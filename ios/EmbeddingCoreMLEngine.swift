// EmbeddingCoreMLEngine.swift
// 一个可直接拷进 iOS 工程的 Core ML 句向量引擎（示例实现）。
//
// 与 Kotlin 侧的协议对应：见 kmp/IosEmbeddingBridge.kt
// - 模型 bge_int8.mlmodelc 随 App 打包（xcrun coremlcompiler compile 产物）
// - 输入固定 (1, 512)，int32；attention_mask 为 0 的补零位置不影响 CLS 向量
// - 输出 sentence_embedding：512 维、已 L2 归一化（CLS pooling 烤进了模型）
//
// 用法：
//   1. 把本文件与 bge_int8.mlmodelc 加入 target
//   2. 启动时 engine = EmbeddingCoreMLEngine(); engine.load()
//   3. engine.embed(textTokens: attentionMask: tokenTypeIds:) 返回 [Float]（512 维）

import Foundation
import CoreML

/// 与 Kotlin 侧 OnnxEngineBridge 等价的 Swift 侧最小协议。
public protocol EmbeddingEngineBridge: AnyObject {
    func doInitModel(modelData: Data)
    func generate(inputIds: Data, attentionMask: Data, tokenTypeIds: Data, seqLen: Int64) -> [Float]
    func close()
}

public final class EmbeddingCoreMLEngine: NSObject, EmbeddingEngineBridge {

    public struct Config {
        public var modelName = "bge_int8"
        public var sequenceLength = 512
        public var outputName = "sentence_embedding"
        public var computeUnits: MLComputeUnits = .all
        public init() {}
    }

    private let config: Config
    private var model: MLModel?

    public init(config: Config = Config()) {
        self.config = config
        super.init()
    }

    // MARK: - 加载

    /// 从主 bundle 加载 .mlmodelc。返回是否加载成功。
    @discardableResult
    public func load() -> Bool {
        if model != nil { return true }
        guard let url = Bundle.main.url(
            forResource: config.modelName,
            withExtension: "mlmodelc"
        ) else {
            print("[CoreML] 找不到 \(config.modelName).mlmodelc")
            return false
        }
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = config.computeUnits
            model = try MLModel(contentsOf: url, configuration: cfg)
            print("[CoreML] 模型加载成功: \(url.lastPathComponent)")
            return true
        } catch {
            print("[CoreML] 模型加载失败: \(error)")
            model = nil
            return false
        }
    }

    // MARK: - 推理

    /// 返回 512 维单位向量；模型未加载或推理失败时返回空数组（调用方必须判空）。
    public func embed(inputIds: [Int], attentionMask: [Int], tokenTypeIds: [Int]) -> [Float] {
        guard let model else {
            print("[CoreML] 模型未加载")
            return []
        }
        let n = min(min(inputIds.count, attentionMask.count), tokenTypeIds.count)
        do {
            let shape: [NSNumber] = [1, NSNumber(value: config.sequenceLength)]
            let ids = try Self.makeInt32Array(inputIds, count: n, shape: shape, padded: config.sequenceLength)
            let mask = try Self.makeInt32Array(attentionMask, count: n, shape: shape, padded: config.sequenceLength)
            let types = try Self.makeInt32Array(tokenTypeIds, count: n, shape: shape, padded: config.sequenceLength)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: ids),
                "attention_mask": MLFeatureValue(multiArray: mask),
                "token_type_ids": MLFeatureValue(multiArray: types),
            ])

            let result = try model.prediction(from: input)
            guard let out = result.featureValue(for: config.outputName)?.multiArrayValue else {
                print("[CoreML] 输出缺失: \(config.outputName)")
                return []
            }
            var v = [Float](repeating: 0, count: out.count)
            for i in 0..<out.count { v[i] = out[i].floatValue }
            return v
        } catch {
            print("[CoreML] 推理失败: \(error)")
            return []
        }
    }

    // MARK: - 协议实现（供 Kotlin/Native 调用）

    public func doInitModel(modelData: Data) {
        // 模型在 bundle 内，Kotlin 传来的字节只是"启动信号"
        _ = modelData
        load()
    }

    public func generate(inputIds: Data, attentionMask: Data, tokenTypeIds: Data, seqLen: Int64) -> [Float] {
        let ids = Self.int64Array(from: inputIds)
        let mask = Self.int64Array(from: attentionMask)
        let types = Self.int64Array(from: tokenTypeIds)
        let n = min(ids.count, Int(seqLen))
        return embed(
            inputIds: Array(ids.prefix(n)),
            attentionMask: Array(mask.prefix(n)),
            tokenTypeIds: Array(types.prefix(n))
        )
    }

    public func close() {
        model = nil
    }

    // MARK: - 工具

    private static func makeInt32Array(
        _ src: [Int], count: Int, shape: [NSNumber], padded: Int
    ) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape, dataType: .int32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        // 防御：count 与 padded 都来自不可信输入，取较小值避免越界写
        let n = min(max(count, 0), padded)
        for i in 0..<n { dst[i] = Int32(truncatingIfNeeded: src[i]) }
        for i in n..<padded { dst[i] = 0 }
        return arr
    }

    private static func int64Array(from data: Data) -> [Int] {
        let n = data.count / MemoryLayout<Int64>.size
        return data.withUnsafeBytes { raw -> [Int] in
            let p = raw.bindMemory(to: Int64.self)
            return (0..<n).map { Int(p[$0]) }
        }
    }
}
