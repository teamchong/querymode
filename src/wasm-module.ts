/**
 * WASM module import — used by DOs in Wrangler modules format.
 * The `.wasm` import returns a compiled WebAssembly.Module.
 */
// @ts-expect-error — Wrangler handles .wasm imports as CompiledWasm modules
import wasmModule from "./wasm/querymode.wasm";

export default wasmModule as WebAssembly.Module;
