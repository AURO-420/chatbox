import { Directory, Filesystem, type FilesystemPlugin } from '@capacitor/filesystem'
import { CapacitorSQLite, SQLiteConnection, type SQLiteDBConnection } from '@capacitor-community/sqlite'
import { embedMany, type EmbeddingModel } from 'ai'
import { v4 as uuidv4 } from 'uuid'
import { isPdfFilePath, isTextFilePath } from '@shared/file-extensions'
import type { CallChatCompletionOptions } from '@shared/models/types'
import {
  SessionSettingsSchema,
  type FileMeta,
  type KnowledgeBase,
  type KnowledgeBaseFile,
  type KnowledgeBaseProviderMode,
  type KnowledgeBaseSearchResult,
} from '@shared/types'
import type { DocumentParserConfig } from '@shared/types/settings'
import { parseKnowledgeBaseModelString } from '@shared/utils/knowledge-base-model-parser'
import { KNOWLEDGE_BASE_MAX_FILE_SIZE, KNOWLEDGE_BASE_MAX_PARSED_CONTENT_SIZE } from '@shared/knowledge-base'
import { currentModelFactory } from '@/adapters/CurrentModelFactory'
import { settingsService } from '@/settings-runtime'
import { parsePdfFileLocally } from '@/packages/pdf-parser'
import type { KnowledgeBaseController } from './interface'

const DATABASE_NAME = 'chatbox-kb.db'
const FILES_DIRECTORY = 'knowledge-base/files'
const CHUNK_SIZE = 1200
const CHUNK_OVERLAP = 150
const EMBEDDING_BATCH_SIZE = 24

type DatabaseRow = Record<string, unknown>

interface EmbeddingCapableModel {
  getTextEmbeddingModel(options: CallChatCompletionOptions): EmbeddingModel
}

interface MobileKnowledgeBaseControllerOptions {
  database?: SQLiteDBConnection
  filesystem?: FilesystemPlugin
  embeddingModelFactory?: (knowledgeBase: KnowledgeBase) => Promise<EmbeddingModel>
  parseFile?: (file: File) => Promise<string>
}

let sharedDatabasePromise: Promise<SQLiteDBConnection> | null = null

function getDatabaseConnection(): Promise<SQLiteDBConnection> {
  if (!sharedDatabasePromise) {
    sharedDatabasePromise = (async () => {
      const sqlite = new SQLiteConnection(CapacitorSQLite)
      try {
        await sqlite.closeConnection(DATABASE_NAME, false)
      } catch {
        // A new install has no previous native connection.
      }
      const database = await sqlite.createConnection(DATABASE_NAME, false, 'no-encryption', 1, false)
      await database.open()
      await initializeSchema(database)
      return database
    })().catch((error) => {
      sharedDatabasePromise = null
      throw error
    })
  }
  return sharedDatabasePromise
}

async function initializeSchema(database: SQLiteDBConnection): Promise<void> {
  await database.execute('PRAGMA foreign_keys = ON')
  await database.execute(
    'CREATE TABLE IF NOT EXISTS knowledge_base (' +
      'id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, embedding_model TEXT NOT NULL, ' +
      'rerank_model TEXT, vision_model TEXT, provider_mode TEXT, document_parser TEXT, created_at INTEGER NOT NULL)'
  )
  await database.execute(
    'CREATE TABLE IF NOT EXISTS kb_file (' +
      'id INTEGER PRIMARY KEY AUTOINCREMENT, kb_id INTEGER NOT NULL, filename TEXT NOT NULL, filepath TEXT NOT NULL, ' +
      'mime_type TEXT NOT NULL, file_size INTEGER NOT NULL DEFAULT 0, chunk_count INTEGER NOT NULL DEFAULT 0, ' +
      'total_chunks INTEGER NOT NULL DEFAULT 0, staged_chunk_count INTEGER NOT NULL DEFAULT 0, ' +
      "status TEXT NOT NULL DEFAULT 'pending', error TEXT, created_at INTEGER NOT NULL, processing_started_at INTEGER, " +
      'pending_content TEXT, content_text TEXT, staging_generation TEXT, parsed_remotely INTEGER NOT NULL DEFAULT 0, ' +
      "parser_type TEXT DEFAULT 'local', use_remote_parsing INTEGER NOT NULL DEFAULT 0, " +
      'FOREIGN KEY (kb_id) REFERENCES knowledge_base(id) ON DELETE CASCADE)'
  )
  await database.execute(
    'CREATE TABLE IF NOT EXISTS kb_chunk (' +
      'id INTEGER PRIMARY KEY AUTOINCREMENT, kb_id INTEGER NOT NULL, file_id INTEGER NOT NULL, ' +
      'filename TEXT NOT NULL, mime_type TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT NOT NULL, ' +
      'FOREIGN KEY (kb_id) REFERENCES knowledge_base(id) ON DELETE CASCADE, ' +
      'FOREIGN KEY (file_id) REFERENCES kb_file(id) ON DELETE CASCADE, UNIQUE(file_id, chunk_index))'
  )
  await database.execute(
    'CREATE TABLE IF NOT EXISTS kb_chunk_staging (' +
      'kb_id INTEGER NOT NULL, file_id INTEGER NOT NULL, generation TEXT NOT NULL, filename TEXT NOT NULL, ' +
      'mime_type TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, embedding TEXT NOT NULL, ' +
      'PRIMARY KEY(file_id, generation, chunk_index))'
  )
  await database.execute('CREATE INDEX IF NOT EXISTS idx_kb_file_kb_created ON kb_file(kb_id, created_at DESC)')
  await database.execute('CREATE INDEX IF NOT EXISTS idx_kb_chunk_kb_file ON kb_chunk(kb_id, file_id, chunk_index)')
  await database.execute(
    "UPDATE kb_file SET status = 'pending', processing_started_at = NULL WHERE status = 'processing'"
  )
}

function toBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192))
  }
  return btoa(binary)
}

function fromBase64(value: string): Uint8Array {
  const binary = atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index)
  return bytes
}

function toTextBase64(value: string): string {
  return toBase64(new TextEncoder().encode(value))
}

function safeFileName(value: string): string {
  return value.replace(/[\\/:*?"<>|\u0000-\u001f]/g, '_').slice(-120) || 'document'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function mapKnowledgeBase(row: DatabaseRow): KnowledgeBase {
  return {
    id: Number(row.id),
    name: String(row.name ?? ''),
    embeddingModel: String(row.embedding_model ?? ''),
    rerankModel: String(row.rerank_model ?? ''),
    visionModel: row.vision_model ? String(row.vision_model) : undefined,
    providerMode: row.provider_mode ? (String(row.provider_mode) as KnowledgeBaseProviderMode) : undefined,
    documentParser: row.document_parser ? (JSON.parse(String(row.document_parser)) as DocumentParserConfig) : undefined,
    createdAt: Number(row.created_at ?? Date.now()),
  }
}

function mapKnowledgeBaseFile(row: DatabaseRow): KnowledgeBaseFile {
  return {
    id: Number(row.id),
    kb_id: Number(row.kb_id),
    filename: String(row.filename ?? ''),
    filepath: String(row.filepath ?? ''),
    mime_type: String(row.mime_type ?? 'application/octet-stream'),
    file_size: Number(row.file_size ?? 0),
    chunk_count: Number(row.chunk_count ?? 0),
    total_chunks: Number(row.total_chunks ?? 0),
    status: String(row.status ?? 'pending'),
    error: row.error ? String(row.error) : '',
    createdAt: Number(row.created_at ?? Date.now()),
    parsed_remotely: Number(row.parsed_remotely ?? 0),
    parser_type: row.parser_type ? (String(row.parser_type) as KnowledgeBaseFile['parser_type']) : 'local',
  }
}

export function chunkKnowledgeBaseText(content: string): string[] {
  const normalized = content.replace(/\r\n/g, '\n').trim()
  if (!normalized) return []
  const chunks: string[] = []
  let start = 0
  while (start < normalized.length) {
    let end = Math.min(start + CHUNK_SIZE, normalized.length)
    if (end < normalized.length) {
      const preferredBreak = normalized.lastIndexOf('\n', end)
      if (preferredBreak > start + Math.floor(CHUNK_SIZE * 0.6)) end = preferredBreak
    }
    const chunk = normalized.slice(start, end).trim()
    if (chunk) chunks.push(chunk)
    if (end >= normalized.length) break
    start = Math.max(start + 1, end - CHUNK_OVERLAP)
  }
  return chunks
}

export function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length === 0 || left.length !== right.length) return 0
  let dot = 0
  let leftNorm = 0
  let rightNorm = 0
  for (let index = 0; index < left.length; index++) {
    dot += left[index] * right[index]
    leftNorm += left[index] * left[index]
    rightNorm += right[index] * right[index]
  }
  if (leftNorm === 0 || rightNorm === 0) return 0
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm))
}

async function parseMobileDocument(file: File): Promise<string> {
  if (isTextFilePath(file.name)) return file.text()
  if (isPdfFilePath(file.name)) return parsePdfFileLocally(file)
  if (/\.(docx|xlsx|pptx)$/i.test(file.name)) {
    if (file.size > 10 * 1024 * 1024) {
      throw new Error('This Office file is larger than the 10 MB mobile parsing limit.')
    }
    const { capacitorExecution } = await import('@/packages/mobile-work-mode/execution-service')
    const parsed = await capacitorExecution.parseDocument(
      file.name,
      toBase64(new Uint8Array(await file.arrayBuffer())),
      new AbortController().signal
    )
    return parsed.text + (parsed.truncated ? '\n[Document text truncated at the local parsing limit.]' : '')
  }
  throw new Error('This file format is not supported by the mobile knowledge base parser yet.')
}

export class MobileKnowledgeBaseController implements KnowledgeBaseController {
  private readonly filesystem: FilesystemPlugin
  private readonly injectedDatabase?: SQLiteDBConnection
  private readonly embeddingModelFactory: (knowledgeBase: KnowledgeBase) => Promise<EmbeddingModel>
  private readonly parseFile: (file: File) => Promise<string>
  private databasePromise: Promise<SQLiteDBConnection> | null = null
  private workerPromise: Promise<void> | null = null
  private readonly embeddingModels = new Map<number, Promise<EmbeddingModel>>()
  private readonly fileLocks = new Map<number, Promise<void>>()

  constructor(options: MobileKnowledgeBaseControllerOptions = {}) {
    this.filesystem = options.filesystem ?? Filesystem
    this.injectedDatabase = options.database
    this.embeddingModelFactory = options.embeddingModelFactory ?? this.createEmbeddingModel.bind(this)
    this.parseFile = options.parseFile ?? parseMobileDocument
    this.requestWorker()
  }

  private async database(): Promise<SQLiteDBConnection> {
    if (this.injectedDatabase) return this.injectedDatabase
    if (!this.databasePromise) this.databasePromise = getDatabaseConnection()
    return this.databasePromise
  }

  private async withFileLock<T>(fileId: number, operation: () => Promise<T>): Promise<T> {
    const previous = this.fileLocks.get(fileId) ?? Promise.resolve()
    let release = () => {}
    const current = new Promise<void>((resolve) => {
      release = resolve
    })
    this.fileLocks.set(fileId, current)
    await previous
    try {
      return await operation()
    } finally {
      release()
      if (this.fileLocks.get(fileId) === current) this.fileLocks.delete(fileId)
    }
  }

  private async createEmbeddingModel(knowledgeBase: KnowledgeBase): Promise<EmbeddingModel> {
    const parsed = parseKnowledgeBaseModelString(knowledgeBase.embeddingModel)
    if (!parsed) throw new Error('The knowledge base embedding model is not configured correctly.')
    const globalSettings = settingsService.getSettings()
    if (!globalSettings.providers?.[parsed.providerId]) {
      throw new Error('The provider configured for this knowledge base is not available. Check Settings → Provider.')
    }
    const sessionSettings = SessionSettingsSchema.parse({
      ...globalSettings,
      provider: parsed.providerId,
      modelId: parsed.modelId,
    })
    const model = await currentModelFactory.createModel(sessionSettings)
    const method = (model as unknown as Partial<EmbeddingCapableModel>).getTextEmbeddingModel
    if (typeof method !== 'function') throw new Error('The selected model does not support text embeddings.')
    return method.call(model, {})
  }

  private async getEmbeddingModel(kbId: number): Promise<EmbeddingModel> {
    let pending = this.embeddingModels.get(kbId)
    if (!pending) {
      pending = (async () => {
        const db = await this.database()
        const result = await db.query('SELECT * FROM knowledge_base WHERE id = ?', [kbId])
        const row = result.values?.[0]
        if (!row) throw new Error('Knowledge base not found.')
        return this.embeddingModelFactory(mapKnowledgeBase(row as DatabaseRow))
      })()
      this.embeddingModels.set(kbId, pending)
      pending.catch(() => this.embeddingModels.delete(kbId))
    }
    return pending
  }

  private async ensureFilesDirectory(): Promise<void> {
    try {
      await this.filesystem.mkdir({ path: FILES_DIRECTORY, directory: Directory.Data, recursive: true })
    } catch {
      // The directory may already exist.
    }
  }

  private async readStoredFile(path: string, filename: string, mimeType: string): Promise<File> {
    const result = await this.filesystem.readFile({ path, directory: Directory.Data })
    const base64 = typeof result.data === 'string' ? result.data : toBase64(new Uint8Array(await result.data.arrayBuffer()))
    return new File([fromBase64(base64)], filename, { type: mimeType })
  }

  private requestWorker(): void {
    if (this.workerPromise) return
    this.workerPromise = this.runPendingFiles().finally(() => {
      this.workerPromise = null
    })
  }

  private async runPendingFiles(): Promise<void> {
    try {
      const db = await this.database()
      while (true) {
        const result = await db.query(
          "SELECT id FROM kb_file WHERE status = 'pending' ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        const row = result.values?.[0]
        if (!row) return
        await this.withFileLock(Number(row.id), () => this.processFileUnlocked(Number(row.id)))
      }
    } catch (error) {
      console.error('[MobileKnowledgeBase] Background indexing stopped:', error)
    }
  }

  private async processFileUnlocked(fileId: number): Promise<void> {
    const db = await this.database()
    const result = await db.query('SELECT * FROM kb_file WHERE id = ?', [fileId])
    const row = result.values?.[0] as DatabaseRow | undefined
    if (!row || String(row.status) !== 'pending') return
    const kbId = Number(row.kb_id)
    const filename = String(row.filename)
    const mimeType = String(row.mime_type)

    try {
      await db.run(
        "UPDATE kb_file SET status = 'processing', processing_started_at = ?, error = NULL WHERE id = ?",
        [Date.now(), fileId]
      )
      let content = typeof row.pending_content === 'string' ? row.pending_content : null
      if (content === null) {
        const file = await this.readStoredFile(String(row.filepath), filename, mimeType)
        content = await this.parseFile(file)
        if (new TextEncoder().encode(content).length > KNOWLEDGE_BASE_MAX_PARSED_CONTENT_SIZE) {
          throw new Error('The extracted document text exceeds the 20 MB knowledge base limit.')
        }
        await db.run(
          'UPDATE kb_file SET pending_content = ?, staging_generation = ? WHERE id = ?',
          [content, uuidv4(), fileId]
        )
      }
      const chunks = chunkKnowledgeBaseText(content)
      if (chunks.length === 0) throw new Error('No text could be extracted from this file.')

      const current = await db.query('SELECT staging_generation FROM kb_file WHERE id = ?', [fileId])
      let generation = String(current.values?.[0]?.staging_generation ?? '')
      if (!generation) {
        generation = uuidv4()
        await db.run('UPDATE kb_file SET staging_generation = ? WHERE id = ?', [generation, fileId])
      }
      await db.run('UPDATE kb_file SET total_chunks = ? WHERE id = ?', [chunks.length, fileId])

      const stagedResult = await db.query(
        'SELECT COUNT(*) AS count FROM kb_chunk_staging WHERE file_id = ? AND generation = ?',
        [fileId, generation]
      )
      let stagedCount = Number(stagedResult.values?.[0]?.count ?? 0)
      if (stagedCount > chunks.length) {
        await db.run('DELETE FROM kb_chunk_staging WHERE file_id = ? AND generation = ?', [fileId, generation])
        stagedCount = 0
      }

      const embeddingModel = await this.getEmbeddingModel(kbId)
      for (let offset = stagedCount; offset < chunks.length; offset += EMBEDDING_BATCH_SIZE) {
        const state = await db.query('SELECT status FROM kb_file WHERE id = ?', [fileId])
        if (String(state.values?.[0]?.status) === 'paused') return
        const batch = chunks.slice(offset, offset + EMBEDDING_BATCH_SIZE)
        const embeddings = await embedMany({
          model: embeddingModel,
          values: batch.map((chunk) => 'filename: ' + filename + '\nchunk:\n' + chunk),
          maxRetries: 0,
        })
        if (embeddings.embeddings.length !== batch.length) {
          throw new Error('The embedding provider returned an incomplete batch.')
        }
        const statements = batch.map((chunk, index) => ({
          statement:
            'INSERT OR REPLACE INTO kb_chunk_staging (kb_id, file_id, generation, filename, mime_type, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
          values: [
            kbId,
            fileId,
            generation,
            filename,
            mimeType,
            offset + index,
            chunk,
            JSON.stringify(embeddings.embeddings[index]),
          ],
        }))
        await db.executeSet(statements, true)
        stagedCount += batch.length
        await db.run('UPDATE kb_file SET staged_chunk_count = ? WHERE id = ?', [stagedCount, fileId])
      }

      const finalState = await db.query('SELECT status FROM kb_file WHERE id = ?', [fileId])
      if (String(finalState.values?.[0]?.status) === 'paused') return
      await db.execute('BEGIN TRANSACTION')
      try {
        await db.run('DELETE FROM kb_chunk WHERE file_id = ?', [fileId])
        await db.run(
          'INSERT INTO kb_chunk (kb_id, file_id, filename, mime_type, chunk_index, text, embedding) ' +
            'SELECT kb_id, file_id, filename, mime_type, chunk_index, text, embedding ' +
            'FROM kb_chunk_staging WHERE file_id = ? AND generation = ?',
          [fileId, generation]
        )
        await db.run(
          "UPDATE kb_file SET content_text = pending_content, pending_content = NULL, staging_generation = NULL, " +
            "staged_chunk_count = 0, chunk_count = ?, total_chunks = ?, status = 'done', " +
            "processing_started_at = NULL, error = NULL, parser_type = 'local' WHERE id = ?",
          [chunks.length, chunks.length, fileId]
        )
        await db.run('DELETE FROM kb_chunk_staging WHERE file_id = ?', [fileId])
        await db.execute('COMMIT')
      } catch (error) {
        await db.execute('ROLLBACK').catch(() => undefined)
        throw error
      }
    } catch (error) {
      await db.run(
        "UPDATE kb_file SET status = 'failed', error = ?, processing_started_at = NULL WHERE id = ?",
        [errorMessage(error), fileId]
      )
    }
  }

  async list(): Promise<KnowledgeBase[]> {
    const db = await this.database()
    const result = await db.query('SELECT * FROM knowledge_base ORDER BY created_at DESC, id DESC')
    return (result.values ?? []).map((row) => mapKnowledgeBase(row as DatabaseRow))
  }

  async create(params: {
    name: string
    embeddingModel: string
    rerankModel: string
    visionModel?: string
    documentParser?: DocumentParserConfig
    providerMode?: KnowledgeBaseProviderMode
  }): Promise<void> {
    const name = params.name.trim()
    if (!name) throw new Error('Knowledge base name is required.')
    if (!params.embeddingModel.trim()) throw new Error('Embedding model is required.')
    const db = await this.database()
    await db.run(
      'INSERT INTO knowledge_base (name, embedding_model, rerank_model, vision_model, provider_mode, document_parser, created_at) ' +
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
      [
        name,
        params.embeddingModel,
        params.rerankModel || null,
        params.visionModel || null,
        params.providerMode || null,
        params.documentParser ? JSON.stringify(params.documentParser) : null,
        Date.now(),
      ]
    )
  }

  async delete(id: number): Promise<void> {
    const db = await this.database()
    const result = await db.query('SELECT filepath FROM kb_file WHERE kb_id = ?', [id])
    for (const row of result.values ?? []) {
      await this.filesystem.deleteFile({ path: String(row.filepath), directory: Directory.Data }).catch(() => undefined)
    }
    await db.execute('BEGIN TRANSACTION')
    try {
      await db.run('DELETE FROM kb_chunk_staging WHERE kb_id = ?', [id])
      await db.run('DELETE FROM kb_chunk WHERE kb_id = ?', [id])
      await db.run('DELETE FROM kb_file WHERE kb_id = ?', [id])
      await db.run('DELETE FROM knowledge_base WHERE id = ?', [id])
      await db.execute('COMMIT')
    } catch (error) {
      await db.execute('ROLLBACK').catch(() => undefined)
      throw error
    }
    this.embeddingModels.delete(id)
  }

  async listFiles(kbId: number): Promise<KnowledgeBaseFile[]> {
    const db = await this.database()
    const result = await db.query('SELECT * FROM kb_file WHERE kb_id = ? ORDER BY created_at DESC, id DESC', [kbId])
    return (result.values ?? []).map((row) => mapKnowledgeBaseFile(row as DatabaseRow))
  }

  async countFiles(kbId: number): Promise<number> {
    const db = await this.database()
    const result = await db.query('SELECT COUNT(*) AS count FROM kb_file WHERE kb_id = ?', [kbId])
    return Number(result.values?.[0]?.count ?? 0)
  }

  async listFilesPaginated(kbId: number, offset = 0, limit = 20): Promise<KnowledgeBaseFile[]> {
    const db = await this.database()
    const safeOffset = Math.max(0, Math.floor(offset))
    const safeLimit = Math.min(100, Math.max(1, Math.floor(limit)))
    const result = await db.query(
      'SELECT * FROM kb_file WHERE kb_id = ? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?',
      [kbId, safeLimit, safeOffset]
    )
    return (result.values ?? []).map((row) => mapKnowledgeBaseFile(row as DatabaseRow))
  }

  async uploadFile(kbId: number, meta: FileMeta, sourceFile?: File): Promise<void> {
    if (!sourceFile) throw new Error('The original file data was not provided to mobile storage.')
    if (sourceFile.size > KNOWLEDGE_BASE_MAX_FILE_SIZE) throw new Error('The file exceeds the 50 MB upload limit.')
    const db = await this.database()
    const kbResult = await db.query('SELECT id FROM knowledge_base WHERE id = ?', [kbId])
    if (!kbResult.values?.[0]) throw new Error('Knowledge base not found.')
    await this.ensureFilesDirectory()
    const storedPath = FILES_DIRECTORY + '/' + uuidv4() + '-' + safeFileName(meta.name || sourceFile.name)
    await this.filesystem.writeFile({
      path: storedPath,
      directory: Directory.Data,
      data: toBase64(new Uint8Array(await sourceFile.arrayBuffer())),
    })
    try {
      await db.run(
        "INSERT INTO kb_file (kb_id, filename, filepath, mime_type, file_size, status, created_at) " +
          "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
        [
          kbId,
          meta.name || sourceFile.name,
          storedPath,
          meta.type || sourceFile.type || 'application/octet-stream',
          sourceFile.size,
          Date.now(),
        ]
      )
    } catch (error) {
      await this.filesystem.deleteFile({ path: storedPath, directory: Directory.Data }).catch(() => undefined)
      throw error
    }
    this.requestWorker()
  }

  async deleteFile(fileId: number): Promise<void> {
    const db = await this.database()
    const result = await db.query('SELECT filepath FROM kb_file WHERE id = ?', [fileId])
    const path = result.values?.[0]?.filepath
    await db.run('DELETE FROM kb_chunk_staging WHERE file_id = ?', [fileId])
    await db.run('DELETE FROM kb_chunk WHERE file_id = ?', [fileId])
    await db.run('DELETE FROM kb_file WHERE id = ?', [fileId])
    if (path) await this.filesystem.deleteFile({ path: String(path), directory: Directory.Data }).catch(() => undefined)
  }

  async retryFile(fileId: number, useRemoteParsing = false): Promise<void> {
    if (useRemoteParsing) throw new Error('Remote parsing retries are not available in the mobile knowledge base yet.')
    const db = await this.database()
    const result = await db.query('SELECT status FROM kb_file WHERE id = ?', [fileId])
    if (!result.values?.[0]) throw new Error('File not found.')
    if (String(result.values[0].status) !== 'failed') throw new Error('Only failed files can be retried.')
    await db.run("UPDATE kb_file SET status = 'pending', error = NULL WHERE id = ?", [fileId])
    this.requestWorker()
  }

  async pauseFile(fileId: number): Promise<void> {
    const db = await this.database()
    const result = await db.query('SELECT status FROM kb_file WHERE id = ?', [fileId])
    if (!result.values?.[0]) throw new Error('File not found.')
    if (String(result.values[0].status) !== 'processing') throw new Error('Only processing files can be paused.')
    await db.run("UPDATE kb_file SET status = 'paused', processing_started_at = NULL WHERE id = ?", [fileId])
  }

  async resumeFile(fileId: number): Promise<void> {
    const db = await this.database()
    const result = await db.query('SELECT status FROM kb_file WHERE id = ?', [fileId])
    if (!result.values?.[0]) throw new Error('File not found.')
    if (String(result.values[0].status) !== 'paused') throw new Error('Only paused files can be resumed.')
    await db.run("UPDATE kb_file SET status = 'pending' WHERE id = ?", [fileId])
    this.requestWorker()
  }

  async search(kbId: number, query: string): Promise<KnowledgeBaseSearchResult[]> {
    if (!query.trim()) return []
    const db = await this.database()
    const model = await this.getEmbeddingModel(kbId)
    const queryEmbedding = await embedMany({ model, values: [query.trim()], maxRetries: 0 })
    const vector = queryEmbedding.embeddings[0]
    if (!vector) return []
    const result = await db.query(
      'SELECT id, file_id, filename, mime_type, chunk_index, text, embedding FROM kb_chunk WHERE kb_id = ?',
      [kbId]
    )
    const scored = (result.values ?? []).map((row) => {
      let embedding: number[] = []
      try {
        embedding = JSON.parse(String(row.embedding)) as number[]
      } catch {
        embedding = []
      }
      return {
        id: Number(row.id),
        score: cosineSimilarity(vector, embedding),
        text: String(row.text ?? ''),
        fileId: Number(row.file_id),
        filename: String(row.filename ?? ''),
        mimeType: String(row.mime_type ?? ''),
        chunkIndex: Number(row.chunk_index ?? 0),
      }
    })
    return scored.sort((left, right) => right.score - left.score).slice(0, 20)
  }

  async update(params: { id: number; name?: string; rerankModel?: string; visionModel?: string }): Promise<void> {
    const updates: string[] = []
    const values: Array<string | number | null> = []
    if (params.name !== undefined) {
      if (!params.name.trim()) throw new Error('Knowledge base name cannot be empty.')
      updates.push('name = ?')
      values.push(params.name.trim())
    }
    if (params.rerankModel !== undefined) {
      updates.push('rerank_model = ?')
      values.push(params.rerankModel || null)
    }
    if (params.visionModel !== undefined) {
      updates.push('vision_model = ?')
      values.push(params.visionModel || null)
    }
    if (!updates.length) return
    values.push(params.id)
    const db = await this.database()
    await db.run('UPDATE knowledge_base SET ' + updates.join(', ') + ' WHERE id = ?', values)
  }

  async getFilesMeta(kbId: number, fileIds: number[]) {
    if (fileIds.length === 0) return []
    const db = await this.database()
    const placeholders = fileIds.map(() => '?').join(',')
    const result = await db.query(
      'SELECT id, kb_id, filename, mime_type, file_size, chunk_count, total_chunks, status, created_at ' +
        'FROM kb_file WHERE kb_id = ? AND id IN (' + placeholders + ')',
      [kbId, ...fileIds]
    )
    return (result.values ?? []).map((row) => ({
      id: Number(row.id),
      kbId: Number(row.kb_id),
      filename: String(row.filename ?? ''),
      mimeType: String(row.mime_type ?? ''),
      fileSize: Number(row.file_size ?? 0),
      chunkCount: Number(row.chunk_count ?? 0),
      totalChunks: Number(row.total_chunks ?? 0),
      status: String(row.status ?? 'pending'),
      createdAt: Number(row.created_at ?? Date.now()),
    }))
  }

  async readFileChunks(kbId: number, chunks: { fileId: number; chunkIndex: number }[]) {
    if (chunks.length === 0) return []
    const db = await this.database()
    const results = await Promise.all(
      chunks.map(async (chunk) => {
        const result = await db.query(
          'SELECT file_id, filename, chunk_index, text FROM kb_chunk WHERE kb_id = ? AND file_id = ? AND chunk_index = ?',
          [kbId, chunk.fileId, chunk.chunkIndex]
        )
        const row = result.values?.[0]
        return row
          ? {
              fileId: Number(row.file_id),
              filename: String(row.filename ?? ''),
              chunkIndex: Number(row.chunk_index),
              text: String(row.text ?? ''),
            }
          : null
      })
    )
    return results.filter((item): item is NonNullable<typeof item> => item !== null)
  }

  async testMineruConnection(_apiToken: string): Promise<{ success: boolean; error?: string }> {
    return { success: false, error: 'MinerU connection testing is not available on mobile.' }
  }

  async readFileContent(fileId: number): Promise<string | null> {
    const db = await this.database()
    const result = await db.query('SELECT * FROM kb_file WHERE id = ?', [fileId])
    const row = result.values?.[0] as DatabaseRow | undefined
    if (!row) return null
    if (typeof row.content_text === 'string') return row.content_text
    if (typeof row.pending_content === 'string') return row.pending_content
    const file = await this.readStoredFile(String(row.filepath), String(row.filename), String(row.mime_type))
    return this.parseFile(file)
  }

  async saveFileContent(fileId: number, content: string): Promise<void> {
    if (new TextEncoder().encode(content).length > KNOWLEDGE_BASE_MAX_PARSED_CONTENT_SIZE) {
      throw new Error('The edited content exceeds the 20 MB knowledge base limit.')
    }
    await this.withFileLock(fileId, async () => {
      const db = await this.database()
      const result = await db.query('SELECT * FROM kb_file WHERE id = ?', [fileId])
      const row = result.values?.[0] as DatabaseRow | undefined
      if (!row) throw new Error('File not found.')
      const generation = uuidv4()
      await db.run('DELETE FROM kb_chunk_staging WHERE file_id = ?', [fileId])
      await db.run(
        "UPDATE kb_file SET pending_content = ?, staging_generation = ?, staged_chunk_count = 0, " +
          "status = 'pending', error = NULL, total_chunks = 0 WHERE id = ?",
        [content, generation, fileId]
      )
      await this.processFileUnlocked(fileId)
      const after = await db.query('SELECT status, error FROM kb_file WHERE id = ?', [fileId])
      if (String(after.values?.[0]?.status) !== 'done') {
        throw new Error(
          String(after.values?.[0]?.error ?? 'The edited file could not be re-indexed. The previous index is still available.')
        )
      }
      if (isTextFilePath(String(row.filename))) {
        await this.filesystem.writeFile({
          path: String(row.filepath),
          directory: Directory.Data,
          data: toTextBase64(content),
        })
      }
    })
  }
}
