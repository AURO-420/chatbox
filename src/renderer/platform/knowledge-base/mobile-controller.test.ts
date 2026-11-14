import { describe, expect, it } from 'vitest'
import { chunkKnowledgeBaseText, cosineSimilarity } from './mobile-controller'

describe('mobile knowledge-base text utilities', () => {
  it('returns no chunks for empty text and normalizes line endings', () => {
    expect(chunkKnowledgeBaseText('  \r\n  ')).toEqual([])
    expect(chunkKnowledgeBaseText('alpha\r\nbeta')).toEqual(['alpha\nbeta'])
  })

  it('splits long text into bounded chunks with overlap', () => {
    const source = 'a'.repeat(2600)
    const chunks = chunkKnowledgeBaseText(source)
    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks.every((chunk) => chunk.length <= 1200)).toBe(true)
    expect(chunks[0].slice(-150)).toBe(chunks[1].slice(0, 150))
  })

  it('prefers paragraph boundaries when a newline is near the chunk limit', () => {
    const source = 'a'.repeat(800) + '\n' + 'b'.repeat(900)
    const chunks = chunkKnowledgeBaseText(source)
    expect(chunks[0]).not.toContain('\n')
    expect(chunks.join('').length).toBeGreaterThan(0)
  })

  it('computes cosine similarity and safely handles empty or mismatched vectors', () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1)
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0)
    expect(cosineSimilarity([], [])).toBe(0)
    expect(cosineSimilarity([1], [1, 2])).toBe(0)
  })
})
