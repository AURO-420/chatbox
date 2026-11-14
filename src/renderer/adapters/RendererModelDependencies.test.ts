import { afterEach, describe, expect, test, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ nativeFetch: vi.fn() }))

vi.mock('@/native/native-fetch', () => ({ nativeFetch: mocks.nativeFetch }))

import { createModelDependencies } from './RendererModelDependencies'

describe('createModelDependencies request transport', () => {
  afterEach(() => {
    mocks.nativeFetch.mockReset()
    vi.unstubAllGlobals()
  })

  test('routes mobile model-list fetches through the native HTTP transport', async () => {
    const response = new Response('{"data":[]}', { status: 200 })
    mocks.nativeFetch.mockResolvedValue(response)
    const rendererFetch = vi.fn()
    vi.stubGlobal('fetch', rendererFetch)

    const dependencies = await createModelDependencies({
      platformInfo: { type: 'mobile', platform: 'android', os: 'android', version: '1.0.0' },
    })

    await expect(dependencies.request.fetchWithOptions('https://provider.example/v1/models')).resolves.toBe(response)
    expect(mocks.nativeFetch).toHaveBeenCalledWith('https://provider.example/v1/models', undefined)
    expect(rendererFetch).not.toHaveBeenCalled()
  })

  test('keeps desktop model-list fetches on the renderer fetch implementation', async () => {
    const response = new Response('{"data":[]}', { status: 200 })
    const rendererFetch = vi.fn().mockResolvedValue(response)
    vi.stubGlobal('fetch', rendererFetch)

    const dependencies = await createModelDependencies({
      platformInfo: { type: 'desktop', platform: 'windows', os: 'windows', version: '1.0.0' },
    })

    await expect(dependencies.request.fetchWithOptions('https://provider.example/v1/models')).resolves.toBe(response)
    expect(rendererFetch).toHaveBeenCalledOnce()
    expect(mocks.nativeFetch).not.toHaveBeenCalled()
  })
})
