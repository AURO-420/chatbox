import { CapacitorSQLite, SQLiteConnection, type SQLiteDBConnection } from '@capacitor-community/sqlite'
import { embedMany, type EmbeddingModel } from 'ai'
import type {
  SessionAttachment,
  SessionAttachmentParent,
  SessionAttachmentQueryPlan,
  SessionAttachmentRagDebugSnapshot,
  SessionAttachmentRagMaintenanceResult,
  SessionAttachmentRagMaintenanceScope,
  SessionAttachmentSearchResult,
} from '@shared/types'
import { planOrphanCleanup } from '@shared/session-attachment-rag/ownership'
import { SessionSettingsSchema } from '@shared/types'
import { parseKnowledgeBaseModelString } from '@shared/utils/knowledge-base-model-parser'
import { buildAttachmentChunks, buildEmbeddedText } from '../../../main/session-attachment-rag/chunking'
import type { SessionAttachmentRagController } from './interface'

const DATABASE_NAME = 'chatbox-session-rag.db'
const EMBEDDING_BATCH_SIZE = 32
const MAX_RECALL_TOP_K = 20
const DEFAULT_FINAL_TOP_K = 8
const MAX_FINAL_TOP_K = 12

type DatabaseRow = Record<string, unknown>

interface MobileSessionAttachmentRagControllerOptions {
  getContent: (storageKey: string) => Promise<string | null>
  embeddingModelFactory?: (modelString: string) => Promise<EmbeddingModel>
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
    'CREATE TABLE IF NOT EXISTS session_attachment (' +
      'id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, message_id TEXT NOT NULL, ' +
      'attachment_storage_key TEXT NOT NULL, filename TEXT NOT NULL, mime_type TEXT NOT NULL, ' +
      'file_size INTEGER NOT NULL DEFAULT 0, token_estimate INTEGER NOT NULL DEFAULT 0, parser_type TEXT, ' +
      "status TEXT NOT NULL DEFAULT 'pending', indexing_stage TEXT, total_chunks INTEGER NOT NULL DEFAULT 0, " +
      'embedded_chunks INTEGER NOT NULL DEFAULT 0, embedding_model TEXT, embedding_dimension INTEGER, error TEXT, ' +
      'created_at INTEGER NOT NULL, processing_started_at INTEGER, completed_at INTEGER)'
  )
  await database.execute(
    'CREATE TABLE IF NOT EXISTS session_attachment_parent (' +
      'id INTEGER PRIMARY KEY AUTOINCREMENT, attachment_id INTEGER NOT NULL, parent_order INTEGER NOT NULL, ' +
      'section_path TEXT, doc_type TEXT, page_start INTEGER, page_end INTEGER, text TEXT NOT NULL, ' +
      'token_estimate INTEGER NOT NULL DEFAULT 0, char_count INTEGER NOT NULL DEFAULT 0, ' +
      'FOREIGN KEY (attachment_id) REFERENCES session_attachment(id) ON DELETE CASCADE)'
  )
  await database.execute(
    'CREATE TABLE IF NOT EXISTS session_attachment_chunk (' +
      'id INTEGER PRIMARY KEY AUTOINCREMENT, attachment_id INTEGER NOT NULL, parent_id INTEGER NOT NULL, ' +
      'chunk_order INTEGER NOT NULL, section_path TEXT, page_start INTEGER, page_end INTEGER, raw_text TEXT NOT NULL, ' +
      'embedded_text TEXT NOT NULL, token_estimate INTEGER NOT NULL DEFAULT 0, embedding TEXT NOT NULL, ' +
      'FOREIGN KEY (attachment_id) REFERENCES session_attachment(id) ON DELETE CASCADE, ' +
      'FOREIGN KEY (parent_id) REFERENCES session_attachment_parent(id) ON DELETE CASCADE)'
  )
  await database.execute('CREATE INDEX IF NOT EXISTS idx_mobile_sa_session ON session_attachment(session_id)')
  await database.execute('CREATE INDEX IF NOT EXISTS idx_mobile_sa_message ON session_attachment(message_id)')
  await database.execute('CREATE INDEX IF NOT EXISTS idx_mobile_sa_status ON session_attachment(status)')
  await database.execute('CREATE INDEX IF NOT EXISTS idx_mobile_sa_parent_attachment ON session_attachment_parent(attachment_id)')
  await database.execute('CREATE INDEX IF NOT EXISTS idx_mobile_sa_chunk_attachment ON session_attachment_chunk(attachment_id)')
  await database.execute(
    "UPDATE session_attachment SET status = 'pending', indexing_stage = 'queued', processing_started_at = NULL WHERE status = 'indexing'"
  )
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function cosineSimilarity(left: number[], right: number[]): number {
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

function mapAttachment(row: DatabaseRow): SessionAttachment {
  const rawStatus = String(row.status ?? 'pending')
  const status = (rawStatus === 'canceled' ? 'failed' : rawStatus) as SessionAttachment['status']
  return {
    id: Number(row.id),
    sessionId: String(row.session_id ?? ''),
    messageId: String(row.message_id ?? ''),
    attachmentStorageKey: String(row.attachment_storage_key ?? ''),
    filename: String(row.filename ?? ''),
    mimeType: String(row.mime_type ?? 'text/plain'),
    fileSize: Number(row.file_size ?? 0),
    tokenEstimate: Number(row.token_estimate ?? 0),
    chunkCount: Number(row.chunk_count ?? 0),
    totalChunks: Number(row.total_chunks ?? 0),
    embeddedChunks: Number(row.embedded_chunks ?? 0),
    embeddingModel: row.embedding_model ? String(row.embedding_model) : undefined,
    embeddingDimension: row.embedding_dimension == null ? undefined : Number(row.embedding_dimension),
    indexingStage: row.indexing_stage ? (String(row.indexing_stage) as SessionAttachment['indexingStage']) : undefined,
    parserType: row.parser_type ? String(row.parser_type) : undefined,
    availability: 'allowed',
    indexStatus: status,
    status,
    error: row.error ? String(row.error) : undefined,
    createdAt: Number(row.created_at ?? Date.now()),
    processingStartedAt: row.processing_started_at == null ? undefined : Number(row.processing_started_at),
    completedAt: row.completed_at == null ? undefined : Number(row.completed_at),
    resumable: false,
  }
}

function normalizePlan(plan?: SessionAttachmentQueryPlan) {
  return {
    recallTopK: Math.max(1, Math.min(plan?.recallTopK ?? MAX_RECALL_TOP_K, MAX_RECALL_TOP_K)),
    finalTopK: Math.max(1, Math.min(plan?.finalTopK ?? DEFAULT_FINAL_TOP_K, MAX_FINAL_TOP_K)),
  }
}

export class MobileSessionAttachmentRagController implements SessionAttachmentRagController {
  private readonly getContent: (storageKey: string) => Promise<string | null>
  private readonly embeddingModelFactory: (modelString: string) => Promise<EmbeddingModel>
  private readonly embeddingModels = new Map<string, Promise<EmbeddingModel>>()
  private databasePromise: Promise<SQLiteDBConnection> | null = null
  private workerPromise: Promise<void> | null = null
  private workerRequested = false

  constructor(options: MobileSessionAttachmentRagControllerOptions) {
    this.getContent = options.getContent
    this.embeddingModelFactory = options.embeddingModelFactory ?? this.createEmbeddingModel.bind(this)
  }

  private database(): Promise<SQLiteDBConnection> {
    if (!this.databasePromise) this.databasePromise = getDatabaseConnection()
    return this.databasePromise
  }

  private async getEmbeddingModel(): Promise<{ model: EmbeddingModel; modelString: string }> {
    const [{ settingsService }] = await Promise.all([import('@/settings-runtime')])
    const settings = settingsService.getSettings()
    const selected = settings.defaultEmbeddingModel
    const modelString =
      selected?.provider && selected?.model
        ? selected.provider + ':' + selected.model
        : settings.licenseKey
          ? 'chatbox-ai:text-embedding-3-small'
          : undefined
    if (!modelString) {
      throw new Error('Session attachment indexing needs a default embedding model. Configure one in Settings → Default Models.')
    }
    let pending = this.embeddingModels.get(modelString)
    if (!pending) {
      pending = this.embeddingModelFactory(modelString)
      this.embeddingModels.set(modelString, pending)
      pending.catch(() => this.embeddingModels.delete(modelString))
    }
    return { model: await pending, modelString }
  }

  private async createEmbeddingModel(modelString: string): Promise<EmbeddingModel> {
    const parsed = parseKnowledgeBaseModelString(modelString)
    if (!parsed) throw new Error('The session attachment embedding model is not configured correctly.')
    const [{ currentModelFactory }, { settingsService }] = await Promise.all([
      import('@/adapters/CurrentModelFactory'),
      import('@/settings-runtime'),
    ])
    const settings = settingsService.getSettings()
    if (!settings.providers?.[parsed.providerId] && parsed.providerId !== 'chatbox-ai') {
      throw new Error('The provider configured for session attachment indexing is unavailable.')
    }
    const sessionSettings = SessionSettingsSchema.parse({
      ...settings,
      provider: parsed.providerId,
      modelId: parsed.modelId,
    })
    const model = await currentModelFactory.createModel(sessionSettings)
    const getTextEmbeddingModel = (model as unknown as { getTextEmbeddingModel?: (options: object) => EmbeddingModel })
      .getTextEmbeddingModel
    if (typeof getTextEmbeddingModel !== 'function') {
      throw new Error('The selected model does not support text embeddings.')
    }
    return getTextEmbeddingModel.call(model, {})
  }

  private requestWorker(): void {
    this.workerRequested = true
    if (this.workerPromise) return
    this.workerPromise = (async () => {
      do {
        this.workerRequested = false
        await this.runPendingAttachments()
      } while (this.workerRequested)
    })().finally(() => {
      this.workerPromise = null
      if (this.workerRequested) this.requestWorker()
    })
  }

  private async runPendingAttachments(): Promise<void> {
    try {
      const db = await this.database()
      while (true) {
        const result = await db.query(
          "SELECT id FROM session_attachment WHERE status = 'pending' ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        const row = result.values?.[0]
        if (!row) return
        await this.processAttachment(Number(row.id))
      }
    } catch (error) {
      console.error('[MobileSessionAttachmentRAG] Background indexing stopped:', error)
    }
  }

  private async processAttachment(attachmentId: number): Promise<void> {
    const db = await this.database()
    const result = await db.query('SELECT * FROM session_attachment WHERE id = ?', [attachmentId])
    const row = result.values?.[0] as DatabaseRow | undefined
    if (!row || String(row.status) !== 'pending') return
    try {
      await db.run(
        "UPDATE session_attachment SET status = 'indexing', indexing_stage = 'chunking', processing_started_at = ?, error = NULL WHERE id = ?",
        [Date.now(), attachmentId]
      )
      const content = await this.getContent(String(row.attachment_storage_key))
      if (!content?.trim()) throw new Error('The parsed attachment text is not available.')
      const chunking = await buildAttachmentChunks(content, String(row.filename))
      if (chunking.parents.length === 0 || chunking.children.length === 0) {
        throw new Error('No searchable text could be extracted from this attachment.')
      }
      const { model, modelString } = await this.getEmbeddingModel()
      await db.run(
        "UPDATE session_attachment SET indexing_stage = 'embedding', total_chunks = ?, embedding_model = ? WHERE id = ?",
        [chunking.children.length, modelString, attachmentId]
      )
      const embeddedRows: Array<{ values: number[]; child: (typeof chunking.children)[number] }> = []
      for (let offset = 0; offset < chunking.children.length; offset += EMBEDDING_BATCH_SIZE) {
        const batch = chunking.children.slice(offset, offset + EMBEDDING_BATCH_SIZE)
        const values = batch.map((child) =>
          buildEmbeddedText({
            filename: String(row.filename),
            sectionPath: child.sectionPath,
            text: child.rawText,
          })
        )
        const response = await embedMany({ model, values, maxRetries: 0 })
        if (response.embeddings.length !== batch.length) {
          throw new Error('The embedding provider returned an incomplete batch.')
        }
        response.embeddings.forEach((embedding, index) => embeddedRows.push({ values: embedding, child: batch[index] }))
        await db.run('UPDATE session_attachment SET embedded_chunks = ? WHERE id = ?', [
          embeddedRows.length,
          attachmentId,
        ])
        const state = await db.query('SELECT status FROM session_attachment WHERE id = ?', [attachmentId])
        if (!state.values?.[0]) return
      }

      await db.execute('BEGIN TRANSACTION')
      try {
        const state = await db.query('SELECT status FROM session_attachment WHERE id = ?', [attachmentId])
        if (!state.values?.[0]) {
          await db.execute('ROLLBACK')
          return
        }
        await db.run('DELETE FROM session_attachment_chunk WHERE attachment_id = ?', [attachmentId])
        await db.run('DELETE FROM session_attachment_parent WHERE attachment_id = ?', [attachmentId])
        const parentIds = new Map<number, number>()
        for (const parent of chunking.parents) {
          await db.run(
            'INSERT INTO session_attachment_parent (attachment_id, parent_order, section_path, text, token_estimate, char_count) VALUES (?, ?, ?, ?, ?, ?)',
            [attachmentId, parent.parentOrder, parent.sectionPath ?? null, parent.text, parent.tokenEstimate, parent.charCount]
          )
          const idResult = await db.query('SELECT last_insert_rowid() AS id')
          parentIds.set(parent.parentOrder, Number(idResult.values?.[0]?.id))
        }
        for (const item of embeddedRows) {
          const parentId = parentIds.get(item.child.parentOrder)
          if (!parentId) throw new Error('Parent block ' + item.child.parentOrder + ' was not stored.')
          const embeddedText = buildEmbeddedText({
            filename: String(row.filename),
            sectionPath: item.child.sectionPath,
            text: item.child.rawText,
          })
          await db.run(
            'INSERT INTO session_attachment_chunk (attachment_id, parent_id, chunk_order, section_path, raw_text, embedded_text, token_estimate, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            [
              attachmentId,
              parentId,
              item.child.chunkOrder,
              item.child.sectionPath ?? null,
              item.child.rawText,
              embeddedText,
              item.child.tokenEstimate,
              JSON.stringify(item.values),
            ]
          )
        }
        await db.run(
          "UPDATE session_attachment SET status = 'ready', indexing_stage = 'ready', total_chunks = ?, embedded_chunks = ?, embedding_dimension = ?, error = NULL, processing_started_at = NULL, completed_at = ? WHERE id = ?",
          [embeddedRows.length, embeddedRows.length, embeddedRows[0]?.values.length ?? 0, Date.now(), attachmentId]
        )
        await db.execute('COMMIT')
      } catch (error) {
        await db.execute('ROLLBACK').catch(() => undefined)
        throw error
      }
    } catch (error) {
      await db.run(
        "UPDATE session_attachment SET status = 'failed', error = ?, processing_started_at = NULL, completed_at = NULL WHERE id = ? AND status != 'canceled'",
        [errorMessage(error), attachmentId]
      )
    }
  }

  async create(params: {
    sessionId: string
    messageId: string
    attachmentStorageKey: string
    filename: string
    mimeType: string
    fileSize: number
    tokenEstimate: number
    parserType?: string
  }): Promise<SessionAttachment> {
    const db = await this.database()
    await db.run(
      'INSERT INTO session_attachment (session_id, message_id, attachment_storage_key, filename, mime_type, file_size, token_estimate, parser_type, status, indexing_stage, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
      [
        params.sessionId,
        params.messageId,
        params.attachmentStorageKey,
        params.filename,
        params.mimeType,
        params.fileSize,
        params.tokenEstimate,
        params.parserType ?? null,
        'pending',
        'queued',
        Date.now(),
      ]
    )
    const idResult = await db.query('SELECT last_insert_rowid() AS id')
    const id = Number(idResult.values?.[0]?.id)
    this.requestWorker()
    return {
      id,
      sessionId: params.sessionId,
      messageId: params.messageId,
      attachmentStorageKey: params.attachmentStorageKey,
      filename: params.filename,
      mimeType: params.mimeType,
      fileSize: params.fileSize,
      tokenEstimate: params.tokenEstimate,
      parserType: params.parserType,
      availability: 'allowed',
      indexStatus: 'pending',
      status: 'pending',
      chunkCount: 0,
      totalChunks: 0,
      embeddedChunks: 0,
      indexingStage: 'queued',
      resumable: false,
      createdAt: Date.now(),
    }
  }

  async getAttachments(ids: number[]): Promise<SessionAttachment[]> {
    if (ids.length === 0) return []
    const db = await this.database()
    const placeholders = ids.map(() => '?').join(',')
    const result = await db.query(
      'SELECT a.*, (SELECT COUNT(*) FROM session_attachment_chunk c WHERE c.attachment_id = a.id) AS chunk_count FROM session_attachment a WHERE a.id IN (' +
        placeholders +
        ") AND a.status != 'canceled'",
      ids
    )
    return (result.values ?? []).map((row) => mapAttachment(row as DatabaseRow))
  }

  async retryAttachment(attachmentId: number): Promise<void> {
    const db = await this.database()
    await db.run(
      "UPDATE session_attachment SET status = 'pending', indexing_stage = 'queued', error = NULL, processing_started_at = NULL WHERE id = ? AND status = 'failed'",
      [attachmentId]
    )
    this.requestWorker()
  }

  async rebindAttachment(params: { attachmentId: number; sessionId: string; messageId: string }): Promise<void> {
    const db = await this.database()
    await db.run('UPDATE session_attachment SET session_id = ?, message_id = ? WHERE id = ?', [
      params.sessionId,
      params.messageId,
      params.attachmentId,
    ])
  }

  async deleteAttachment(attachmentId: number): Promise<void> {
    const db = await this.database()
    await db.run('DELETE FROM session_attachment WHERE id = ?', [attachmentId])
  }

  async deleteMessageAttachments(messageId: string): Promise<number[]> {
    const db = await this.database()
    const result = await db.query('SELECT id FROM session_attachment WHERE message_id = ?', [messageId])
    const ids = (result.values ?? []).map((row) => Number(row.id))
    await db.run('DELETE FROM session_attachment WHERE message_id = ?', [messageId])
    return ids
  }

  async deleteSessionAttachments(sessionId: string): Promise<number[]> {
    const db = await this.database()
    const result = await db.query('SELECT id FROM session_attachment WHERE session_id = ?', [sessionId])
    const ids = (result.values ?? []).map((row) => Number(row.id))
    await db.run('DELETE FROM session_attachment WHERE session_id = ?', [sessionId])
    return ids
  }

  async cleanupOrphans(params: { sessionIds: string[]; messageIds: string[] }): Promise<number[]> {
    return this.cleanupWithScope({
      sessionIds: params.sessionIds,
      messageIds: params.messageIds,
      attachmentReferences: [],
    })
  }

  private async cleanupWithScope(scope: SessionAttachmentRagMaintenanceScope): Promise<number[]> {
    const db = await this.database()
    const result = await db.query('SELECT id, session_id, message_id FROM session_attachment')
    const plan = planOrphanCleanup(
      (result.values ?? []).map((row) => ({
        id: Number(row.id),
        sessionId: String(row.session_id),
        messageId: String(row.message_id),
      })),
      scope
    )
    for (const repair of plan.repairs) {
      await this.rebindAttachment(repair)
    }
    if (plan.deleteIds.length > 0) {
      const placeholders = plan.deleteIds.map(() => '?').join(',')
      await db.run('DELETE FROM session_attachment WHERE id IN (' + placeholders + ')', plan.deleteIds)
    }
    return plan.deleteIds
  }

  async getDebugSnapshot(): Promise<SessionAttachmentRagDebugSnapshot> {
    const db = await this.database()
    const [attachments, parents, chunks, statusRows, recent] = await Promise.all([
      db.query('SELECT COUNT(*) AS count FROM session_attachment'),
      db.query('SELECT COUNT(*) AS count FROM session_attachment_parent'),
      db.query('SELECT COUNT(*) AS count FROM session_attachment_chunk'),
      db.query('SELECT status, COUNT(*) AS count FROM session_attachment GROUP BY status'),
      db.query(
        "SELECT a.*, (SELECT COUNT(*) FROM session_attachment_chunk c WHERE c.attachment_id = a.id) AS chunk_count FROM session_attachment a WHERE a.status != 'canceled' ORDER BY a.id DESC LIMIT 20"
      ),
    ])
    const statusCounts = { pending: 0, indexing: 0, ready: 0, failed: 0 }
    for (const row of statusRows.values ?? []) {
      const status = String(row.status) as keyof typeof statusCounts
      if (status in statusCounts) statusCounts[status] = Number(row.count ?? 0)
    }
    return {
      dbPath: 'mobile://chatbox-session-rag.db',
      dbSizeBytes: 0,
      vectorDbPath: 'SQLite embedding column',
      vectorDbSizeBytes: 0,
      attachmentCount: Number(attachments.values?.[0]?.count ?? 0),
      parentCount: Number(parents.values?.[0]?.count ?? 0),
      chunkCount: Number(chunks.values?.[0]?.count ?? 0),
      vectorIndexNames: [],
      statusCounts,
      recentAttachments: (recent.values ?? []).map((row) => {
        const attachment = mapAttachment(row as DatabaseRow)
        return {
          id: attachment.id,
          sessionId: attachment.sessionId,
          messageId: attachment.messageId,
          filename: attachment.filename,
          parserType: attachment.parserType,
          status: attachment.status,
          chunkCount: attachment.chunkCount ?? 0,
          error: attachment.error,
          createdAt: attachment.createdAt,
          processingStartedAt: attachment.processingStartedAt,
          completedAt: attachment.completedAt,
        }
      }),
    }
  }

  async clearAll(): Promise<number> {
    const db = await this.database()
    const result = await db.query('SELECT COUNT(*) AS count FROM session_attachment')
    const count = Number(result.values?.[0]?.count ?? 0)
    await db.run('DELETE FROM session_attachment')
    return count
  }

  async runMaintenance(scope: SessionAttachmentRagMaintenanceScope): Promise<SessionAttachmentRagMaintenanceResult> {
    const db = await this.database()
    const interrupted = await db.query("SELECT COUNT(*) AS count FROM session_attachment WHERE status = 'indexing'")
    const interruptedFailedCount = Number(interrupted.values?.[0]?.count ?? 0)
    await db.run(
      "UPDATE session_attachment SET status = 'pending', indexing_stage = 'queued', processing_started_at = NULL WHERE status = 'indexing'"
    )
    const canceled = await db.query("SELECT COUNT(*) AS count FROM session_attachment WHERE status = 'canceled'")
    const canceledPurgedCount = Number(canceled.values?.[0]?.count ?? 0)
    await db.run("DELETE FROM session_attachment WHERE status = 'canceled'")
    const orphanDeletedIds = await this.cleanupWithScope(scope)
    this.requestWorker()
    return { interruptedFailedCount, canceledPurgedCount, orphanDeletedIds }
  }

  async query(params: {
    attachmentIds: number[]
    query: string
    plan: SessionAttachmentQueryPlan
  }): Promise<SessionAttachmentSearchResult[]> {
    const attachmentIds = [...new Set(params.attachmentIds.filter((id) => Number.isFinite(id)))]
    if (attachmentIds.length === 0 || !params.query.trim()) return []
    const db = await this.database()
    const placeholders = attachmentIds.map(() => '?').join(',')
    const ready = await db.query(
      "SELECT id FROM session_attachment WHERE id IN (" + placeholders + ") AND status = 'ready'",
      attachmentIds
    )
    const readyIds = (ready.values ?? []).map((row) => Number(row.id))
    if (readyIds.length === 0) return []
    const { model } = await this.getEmbeddingModel()
    const queryResult = await embedMany({ model, values: [params.query.trim()], maxRetries: 0 })
    const queryVector = queryResult.embeddings[0]
    if (!queryVector) return []
    const readyPlaceholders = readyIds.map(() => '?').join(',')
    const chunks = await db.query(
      'SELECT c.attachment_id, c.parent_id, c.chunk_order, c.section_path, c.raw_text, c.embedding, a.filename ' +
        'FROM session_attachment_chunk c JOIN session_attachment a ON a.id = c.attachment_id ' +
        'WHERE c.attachment_id IN (' +
        readyPlaceholders +
        ')',
      readyIds
    )
    const matches: SessionAttachmentSearchResult[] = []
    for (const row of chunks.values ?? []) {
      let vector: number[] = []
      try {
        vector = JSON.parse(String(row.embedding)) as number[]
      } catch {
        vector = []
      }
      const score = cosineSimilarity(queryVector, vector)
      if (score <= 0) continue
      matches.push({
        attachmentId: Number(row.attachment_id),
        parentId: Number(row.parent_id),
        filename: String(row.filename ?? ''),
        sectionPath: row.section_path ? String(row.section_path) : undefined,
        chunkOrder: Number(row.chunk_order),
        text: String(row.raw_text ?? ''),
        score,
      })
    }
    matches.sort((left, right) => right.score - left.score)
    const { recallTopK, finalTopK } = normalizePlan(params.plan)
    const recalled = matches.slice(0, recallTopK)
    const deduped = new Map<number, SessionAttachmentSearchResult>()
    for (const match of recalled) {
      if (!deduped.has(match.parentId)) deduped.set(match.parentId, match)
    }
    return Array.from(deduped.values()).slice(0, finalTopK)
  }

  async readParents(params: { parentIds: number[]; attachmentIds: number[] }): Promise<SessionAttachmentParent[]> {
    if (params.parentIds.length === 0 || params.attachmentIds.length === 0) return []
    const db = await this.database()
    const parentPlaceholders = params.parentIds.map(() => '?').join(',')
    const attachmentPlaceholders = params.attachmentIds.map(() => '?').join(',')
    const result = await db.query(
      'SELECT p.*, a.filename FROM session_attachment_parent p ' +
        'JOIN session_attachment a ON a.id = p.attachment_id ' +
        'WHERE p.id IN (' +
        parentPlaceholders +
        ') AND p.attachment_id IN (' +
        attachmentPlaceholders +
        ')',
      [...params.parentIds, ...params.attachmentIds]
    )
    return (result.values ?? []).map((row) => ({
      id: Number(row.id),
      attachmentId: Number(row.attachment_id),
      filename: String(row.filename ?? ''),
      sectionPath: row.section_path ? String(row.section_path) : undefined,
      docType: row.doc_type ? String(row.doc_type) : undefined,
      pageStart: row.page_start == null ? undefined : Number(row.page_start),
      pageEnd: row.page_end == null ? undefined : Number(row.page_end),
      parentOrder: Number(row.parent_order),
      text: String(row.text ?? ''),
      tokenEstimate: Number(row.token_estimate ?? 0),
      charCount: Number(row.char_count ?? 0),
    }))
  }
}
