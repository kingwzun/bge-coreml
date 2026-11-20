// IosEmbeddingBridge.kt
// Kotlin/Native 侧的桥接层：iOS 的推理实现由 Swift 注入，Kotlin 只保留分词与调用逻辑。
//
// 配套 Swift 实现见 ios/EmbeddingCoreMLEngine.swift
// 注意：Kotlin 的 initModel(...) 在 Swift 侧会被当作初始化器家族，
// 实现时必须写成 doInitModel(modelData:)，否则协议无法被满足。

package com.example.embedding   // 替换为你自己的包名

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.Foundation.NSData
import platform.Foundation.create

/** Swift 侧需要实现的接口。 */
interface EmbeddingEngineBridge {
    fun initModel(modelData: NSData)
    fun generate(inputIds: NSData, attentionMask: NSData, tokenTypeIds: NSData, seqLen: Long): FloatArray
    fun close()
}

/** 全局持有 Swift 注入的实例（在 App 启动时赋值）。 */
object IosEmbeddingProvider {
    var bridge: EmbeddingEngineBridge? = null
}

/** 调用侧：维度校验是必须的，否则空向量会被写进索引且永不重建。 */
suspend fun embedOrThrow(text: String, tokenizer: BertTokenizedInput, dimension: Int): FloatArray {
    val bridge = IosEmbeddingProvider.bridge ?: throw IllegalStateException("引擎未加载")
    val vector = bridge.generate(
        inputIds = tokenizer.inputIds.toNSData(),
        attentionMask = tokenizer.attentionMask.toNSData(),
        tokenTypeIds = tokenizer.tokenTypeIds.toNSData(),
        seqLen = tokenizer.inputIds.size.toLong(),
    )
    if (vector.size != dimension) {
        throw IllegalStateException("推理结果异常：期望 $dimension 维，实际 ${vector.size}")
    }
    return vector
}

/** 示例数据载体，实际替换为项目里的分词输出类型。 */
class BertTokenizedInput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val tokenTypeIds: LongArray,
)

// 👇 LongArray/ByteArray → NSData
@OptIn(ExperimentalForeignApi::class)
internal fun LongArray.toNSData(): NSData = usePinned {
    NSData.create(bytes = it.addressOf(0), length = (size * 8).toULong())
}

@OptIn(ExperimentalForeignApi::class)
internal fun ByteArray.toNSData(): NSData =
    if (isEmpty()) NSData() else usePinned {
        NSData.create(bytes = it.addressOf(0), length = size.toULong())
    }
