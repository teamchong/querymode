/**
 * Pure TypeScript HNSW (Hierarchical Navigable Small World) graph
 * for approximate nearest neighbor search.
 *
 * Phase 6 of QueryMode vector search pipeline.
 *
 * References:
 *   Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
 *   using Hierarchical Navigable Small World graphs", 2016.
 */

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

/** Cosine distance: 1 - cosine_similarity(a, b). Range [0, 2]. */
export function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 1;
  return 1 - dot / denom;
}

/** Squared L2 (Euclidean) distance. */
export function l2DistanceSq(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

/** Negative dot product distance (lower = more similar). */
export function dotDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return -dot;
}

type DistanceFn = (a: Float32Array, b: Float32Array) => number;

function getDistanceFn(metric: "cosine" | "l2" | "dot"): DistanceFn {
  switch (metric) {
    case "cosine": return cosineDistance;
    case "l2": return l2DistanceSq;
    case "dot": return dotDistance;
  }
}

// ---------------------------------------------------------------------------
// Min-heap (priority queue) — used for candidate lists
// ---------------------------------------------------------------------------

interface HeapItem {
  dist: number;
  id: number;
}

/** Min-heap ordered by dist (smallest distance at top). */
class MinHeap {
  private data: HeapItem[] = [];

  get length(): number { return this.data.length; }

  peek(): HeapItem | undefined { return this.data[0]; }

  push(item: HeapItem): void {
    this.data.push(item);
    this._bubbleUp(this.data.length - 1);
  }

  pop(): HeapItem | undefined {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0 && last !== undefined) {
      this.data[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  toArray(): HeapItem[] { return this.data.slice(); }

  private _bubbleUp(i: number): void {
    const d = this.data;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (d[i].dist >= d[parent].dist) break;
      [d[i], d[parent]] = [d[parent], d[i]];
      i = parent;
    }
  }

  private _sinkDown(i: number): void {
    const d = this.data;
    const n = d.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      if (l < n && d[l].dist < d[smallest].dist) smallest = l;
      if (r < n && d[r].dist < d[smallest].dist) smallest = r;
      if (smallest === i) break;
      [d[i], d[smallest]] = [d[smallest], d[i]];
      i = smallest;
    }
  }
}

/** Max-heap ordered by dist (largest distance at top). */
class MaxHeap {
  private data: HeapItem[] = [];

  get length(): number { return this.data.length; }

  peek(): HeapItem | undefined { return this.data[0]; }

  push(item: HeapItem): void {
    this.data.push(item);
    this._bubbleUp(this.data.length - 1);
  }

  pop(): HeapItem | undefined {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0 && last !== undefined) {
      this.data[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  toArray(): HeapItem[] { return this.data.slice(); }

  private _bubbleUp(i: number): void {
    const d = this.data;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (d[i].dist <= d[parent].dist) break;
      [d[i], d[parent]] = [d[parent], d[i]];
      i = parent;
    }
  }

  private _sinkDown(i: number): void {
    const d = this.data;
    const n = d.length;
    while (true) {
      let largest = i;
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      if (l < n && d[l].dist > d[largest].dist) largest = l;
      if (r < n && d[r].dist > d[largest].dist) largest = r;
      if (largest === i) break;
      [d[i], d[largest]] = [d[largest], d[i]];
      i = largest;
    }
  }
}

// ---------------------------------------------------------------------------
// HNSW Index
// ---------------------------------------------------------------------------

export interface HnswOptions {
  dim: number;
  metric?: "cosine" | "l2" | "dot";
  /** Max connections per layer (default 16). */
  M?: number;
  /** Construction beam width (default 200). */
  efConstruction?: number;
}

/** Magic bytes for serialization: "HNSW" */
const HNSW_MAGIC = 0x57534e48; // little-endian for "HNSW"

/** Metric enum for serialization. */
const METRIC_BYTE: Record<string, number> = { cosine: 0, l2: 1, dot: 2 };
const BYTE_METRIC: Record<number, "cosine" | "l2" | "dot"> = { 0: "cosine", 1: "l2", 2: "dot" };

/**
 * HNSW (Hierarchical Navigable Small World) graph index for approximate
 * nearest neighbor search. Pure TypeScript, zero dependencies.
 *
 * Typical usage:
 * ```ts
 * const idx = new HnswIndex({ dim: 128, metric: "cosine" });
 * idx.add(0, vec0);
 * idx.add(1, vec1);
 * const { indices, scores } = idx.search(queryVec, 10);
 * ```
 */
export class HnswIndex {
  private readonly dim: number;
  private readonly metric: "cosine" | "l2" | "dot";
  private readonly M: number;
  private readonly Mmax0: number; // max connections at layer 0 (2 * M)
  private readonly efConstruction: number;
  private readonly mL: number; // normalization factor for level generation = 1 / ln(M)
  private readonly distFn: DistanceFn;

  /** Vectors stored contiguously: vectors[id] = Float32Array of length dim. */
  private vectors: Float32Array[] = [];

  /**
   * Adjacency lists per layer.
   * graph[level] is a Map from nodeId to array of neighbor nodeIds.
   */
  private graph: Map<number, number[]>[] = [];

  /** Maximum level currently in the graph. -1 means empty. */
  private maxLevel = -1;

  /** Entry point node id. -1 means empty. */
  private entryPoint = -1;

  /** Number of vectors inserted. */
  private _size = 0;

  /** The level assigned to each node. */
  private nodeLevels: number[] = [];

  constructor(opts: HnswOptions) {
    this.dim = opts.dim;
    this.metric = opts.metric ?? "cosine";
    this.M = opts.M ?? 16;
    this.Mmax0 = this.M * 2;
    this.efConstruction = opts.efConstruction ?? 200;
    this.mL = 1 / Math.log(this.M);
    this.distFn = getDistanceFn(this.metric);
  }

  /** Number of vectors in the index. */
  get size(): number { return this._size; }

  // -----------------------------------------------------------------------
  // Random level assignment
  // -----------------------------------------------------------------------

  private randomLevel(): number {
    // Geometric distribution: floor(-ln(uniform) * mL)
    return Math.floor(-Math.log(Math.random()) * this.mL);
  }

  // -----------------------------------------------------------------------
  // Core graph operations
  // -----------------------------------------------------------------------

  private ensureLevel(level: number): void {
    while (this.graph.length <= level) {
      this.graph.push(new Map());
    }
  }

  private getNeighbors(level: number, nodeId: number): number[] {
    if (level >= this.graph.length) return [];
    return this.graph[level].get(nodeId) ?? [];
  }

  private setNeighbors(level: number, nodeId: number, neighbors: number[]): void {
    this.ensureLevel(level);
    this.graph[level].set(nodeId, neighbors);
  }

  // -----------------------------------------------------------------------
  // Search layer — greedy beam search at a single layer
  // -----------------------------------------------------------------------

  /**
   * Search a single layer starting from entryPoints, returning ef nearest
   * neighbors. Returns a max-heap of (dist, id) pairs.
   */
  private searchLayer(
    query: Float32Array,
    entryPoints: HeapItem[],
    ef: number,
    level: number,
  ): HeapItem[] {
    const visited = new Set<number>();
    const candidates = new MinHeap(); // closest unvisited
    const results = new MaxHeap();    // best ef results (max-heap so we can trim)

    for (const ep of entryPoints) {
      visited.add(ep.id);
      candidates.push(ep);
      results.push(ep);
    }

    while (candidates.length > 0) {
      const closest = candidates.pop()!;
      const farthestResult = results.peek()!;

      // If closest candidate is farther than our worst result, stop
      if (closest.dist > farthestResult.dist) break;

      const neighbors = this.getNeighbors(level, closest.id);
      for (const neighborId of neighbors) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);

        const dist = this.distFn(query, this.vectors[neighborId]);
        const worstResult = results.peek()!;

        if (results.length < ef || dist < worstResult.dist) {
          candidates.push({ dist, id: neighborId });
          results.push({ dist, id: neighborId });
          if (results.length > ef) {
            results.pop(); // evict the farthest
          }
        }
      }
    }

    return results.toArray();
  }

  /**
   * Select M closest neighbors using the simple heuristic.
   * Sorts by distance and takes the first M.
   */
  private selectNeighbors(candidates: HeapItem[], M: number): number[] {
    candidates.sort((a, b) => a.dist - b.dist);
    const result: number[] = [];
    for (let i = 0; i < Math.min(candidates.length, M); i++) {
      result.push(candidates[i].id);
    }
    return result;
  }

  // -----------------------------------------------------------------------
  // Add
  // -----------------------------------------------------------------------

  /** Add a vector to the index. The id must equal the current size (sequential). */
  add(id: number, vector: Float32Array): void {
    if (vector.length !== this.dim) {
      throw new Error(`Vector dimension mismatch: expected ${this.dim}, got ${vector.length}`);
    }

    // Store vector
    if (id !== this._size) {
      throw new Error(`IDs must be sequential. Expected ${this._size}, got ${id}`);
    }
    this.vectors.push(vector);

    const nodeLevel = this.randomLevel();
    this.nodeLevels.push(nodeLevel);
    this.ensureLevel(nodeLevel);
    this._size++;

    // First node: just set as entry point
    if (this._size === 1) {
      this.entryPoint = id;
      this.maxLevel = nodeLevel;
      // Initialize empty neighbor lists at all levels
      for (let l = 0; l <= nodeLevel; l++) {
        this.setNeighbors(l, id, []);
      }
      return;
    }

    let ep = [{ dist: this.distFn(vector, this.vectors[this.entryPoint]), id: this.entryPoint }];

    // Traverse from top layer down to nodeLevel + 1 (greedy, ef=1)
    for (let l = this.maxLevel; l > nodeLevel; l--) {
      const results = this.searchLayer(vector, ep, 1, l);
      // Pick the closest as new entry point
      results.sort((a, b) => a.dist - b.dist);
      ep = [results[0]];
    }

    // For levels nodeLevel down to 0, do full ef-construction search and connect
    for (let l = Math.min(nodeLevel, this.maxLevel); l >= 0; l--) {
      const results = this.searchLayer(vector, ep, this.efConstruction, l);
      const Mcur = l === 0 ? this.Mmax0 : this.M;

      // Select neighbors for the new node
      const neighbors = this.selectNeighbors(results, Mcur);
      this.setNeighbors(l, id, neighbors);

      // Add bidirectional connections
      for (const neighborId of neighbors) {
        const nNeighbors = this.getNeighbors(l, neighborId);
        nNeighbors.push(id);

        if (nNeighbors.length > Mcur) {
          // Shrink: recompute neighbor set
          const nCandidates = nNeighbors.map(nid => ({
            dist: this.distFn(this.vectors[neighborId], this.vectors[nid]),
            id: nid,
          }));
          const pruned = this.selectNeighbors(nCandidates, Mcur);
          this.setNeighbors(l, neighborId, pruned);
        } else {
          this.setNeighbors(l, neighborId, nNeighbors);
        }
      }

      // Update entry points for next layer down
      results.sort((a, b) => a.dist - b.dist);
      ep = [results[0]];
    }

    // Update global entry point if new node is at a higher level
    if (nodeLevel > this.maxLevel) {
      this.entryPoint = id;
      this.maxLevel = nodeLevel;
    }
  }

  /** Batch add vectors from a contiguous Float32Array. */
  addBatch(vectors: Float32Array, dim: number): void {
    if (vectors.length % dim !== 0) {
      throw new Error(`Vector buffer length ${vectors.length} is not divisible by dim ${dim}`);
    }
    const count = vectors.length / dim;
    const startId = this._size;
    for (let i = 0; i < count; i++) {
      const vec = vectors.subarray(i * dim, (i + 1) * dim);
      // subarray shares the underlying buffer — copy for safe storage
      this.add(startId + i, new Float32Array(vec));
    }
  }

  // -----------------------------------------------------------------------
  // Search
  // -----------------------------------------------------------------------

  /** Search for nearest neighbors. Returns topK results sorted by distance (ascending). */
  search(
    query: Float32Array,
    topK: number,
    efSearch?: number,
  ): { indices: Uint32Array; scores: Float32Array } {
    if (this._size === 0) {
      return { indices: new Uint32Array(0), scores: new Float32Array(0) };
    }

    const ef = Math.max(efSearch ?? topK, topK);

    let ep = [{ dist: this.distFn(query, this.vectors[this.entryPoint]), id: this.entryPoint }];

    // Greedy descent from top layer to layer 1
    for (let l = this.maxLevel; l >= 1; l--) {
      const results = this.searchLayer(query, ep, 1, l);
      results.sort((a, b) => a.dist - b.dist);
      ep = [results[0]];
    }

    // Full search at layer 0
    const results = this.searchLayer(query, ep, ef, 0);

    // Sort by distance, take topK
    results.sort((a, b) => a.dist - b.dist);
    const k = Math.min(topK, results.length);
    const indices = new Uint32Array(k);
    const scores = new Float32Array(k);
    for (let i = 0; i < k; i++) {
      indices[i] = results[i].id;
      scores[i] = results[i].dist;
    }

    return { indices, scores };
  }

  // -----------------------------------------------------------------------
  // Serialization
  // -----------------------------------------------------------------------

  /**
   * Serialize the index to a compact binary format.
   *
   * Layout:
   *   Header (25 bytes):
   *     magic: u32 ("HNSW")
   *     dim: u32
   *     M: u32
   *     maxLevel: u32
   *     entryPoint: u32
   *     size: u32
   *     metric: u8
   *   Vectors: size * dim * 4 bytes (float32)
   *   Node levels: size * u32
   *   Per level (maxLevel + 1 levels):
   *     nodeCount: u32
   *     For each node in the level:
   *       nodeId: u32
   *       neighborCount: u32
   *       neighbor_ids: u32[neighborCount]
   */
  serialize(): ArrayBuffer {
    // Calculate total size
    let totalSize = 25; // header
    totalSize += this._size * this.dim * 4; // vectors
    totalSize += this._size * 4; // node levels

    const levelCount = this.maxLevel + 1;
    for (let l = 0; l < levelCount; l++) {
      const map = l < this.graph.length ? this.graph[l] : new Map();
      totalSize += 4; // nodeCount
      for (const [, neighbors] of map) {
        totalSize += 4 + 4 + neighbors.length * 4; // nodeId + neighborCount + neighbors
      }
    }

    const buf = new ArrayBuffer(totalSize);
    const view = new DataView(buf);
    const f32 = new Float32Array(buf);
    let offset = 0;

    // Header
    view.setUint32(offset, HNSW_MAGIC, true); offset += 4;
    view.setUint32(offset, this.dim, true); offset += 4;
    view.setUint32(offset, this.M, true); offset += 4;
    view.setUint32(offset, this.maxLevel >= 0 ? this.maxLevel : 0, true); offset += 4;
    view.setUint32(offset, this.entryPoint >= 0 ? this.entryPoint : 0, true); offset += 4;
    view.setUint32(offset, this._size, true); offset += 4;
    view.setUint8(offset, METRIC_BYTE[this.metric] ?? 0); offset += 1;

    // Vectors
    for (let i = 0; i < this._size; i++) {
      const vec = this.vectors[i];
      for (let d = 0; d < this.dim; d++) {
        view.setFloat32(offset, vec[d], true);
        offset += 4;
      }
    }

    // Node levels
    for (let i = 0; i < this._size; i++) {
      view.setUint32(offset, this.nodeLevels[i], true);
      offset += 4;
    }

    // Per level adjacency
    for (let l = 0; l < levelCount; l++) {
      const map = l < this.graph.length ? this.graph[l] : new Map<number, number[]>();
      view.setUint32(offset, map.size, true); offset += 4;
      for (const [nodeId, neighbors] of map) {
        view.setUint32(offset, nodeId, true); offset += 4;
        view.setUint32(offset, neighbors.length, true); offset += 4;
        for (const nid of neighbors) {
          view.setUint32(offset, nid, true);
          offset += 4;
        }
      }
    }

    return buf;
  }

  /** Deserialize an index from binary produced by serialize(). */
  static deserialize(data: ArrayBuffer, metric?: "cosine" | "l2" | "dot"): HnswIndex {
    const view = new DataView(data);
    let offset = 0;

    // Header
    const magic = view.getUint32(offset, true); offset += 4;
    if (magic !== HNSW_MAGIC) {
      throw new Error(`Invalid HNSW magic: 0x${magic.toString(16)}, expected 0x${HNSW_MAGIC.toString(16)}`);
    }

    const dim = view.getUint32(offset, true); offset += 4;
    const M = view.getUint32(offset, true); offset += 4;
    const maxLevel = view.getUint32(offset, true); offset += 4;
    const entryPoint = view.getUint32(offset, true); offset += 4;
    const size = view.getUint32(offset, true); offset += 4;
    const metricByte = view.getUint8(offset); offset += 1;

    const resolvedMetric = metric ?? BYTE_METRIC[metricByte] ?? "cosine";

    const idx = new HnswIndex({ dim, metric: resolvedMetric, M });
    idx.maxLevel = size > 0 ? maxLevel : -1;
    idx.entryPoint = size > 0 ? entryPoint : -1;

    // Vectors
    for (let i = 0; i < size; i++) {
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        vec[d] = view.getFloat32(offset, true);
        offset += 4;
      }
      idx.vectors.push(vec);
    }
    idx._size = size;

    // Node levels
    for (let i = 0; i < size; i++) {
      idx.nodeLevels.push(view.getUint32(offset, true));
      offset += 4;
    }

    // Per level adjacency
    const levelCount = size > 0 ? maxLevel + 1 : 0;
    for (let l = 0; l < levelCount; l++) {
      idx.ensureLevel(l);
      const nodeCount = view.getUint32(offset, true); offset += 4;
      for (let n = 0; n < nodeCount; n++) {
        const nodeId = view.getUint32(offset, true); offset += 4;
        const neighborCount = view.getUint32(offset, true); offset += 4;
        const neighbors: number[] = [];
        for (let k = 0; k < neighborCount; k++) {
          neighbors.push(view.getUint32(offset, true));
          offset += 4;
        }
        idx.graph[l].set(nodeId, neighbors);
      }
    }

    return idx;
  }
}
