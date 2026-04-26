import { BENCHMARK_SUITE_ID, BENCHMARK_SUITE_VERSION } from "./types/api";
import type { Basis, BenchmarkRequest, DecoderName } from "./types/api";

export function buildBenchmarkRequest(params: {
  distances: number[];
  basis: Basis;
  noiseP: number;
  shots: number;
  seed: number;
  decoders: DecoderName[];
}): BenchmarkRequest {
  return {
    suite_id: BENCHMARK_SUITE_ID,
    suite_version: BENCHMARK_SUITE_VERSION,
    distances: [...new Set(params.distances)].sort((a, b) => a - b),
    rounds: null,
    basis: params.basis,
    noise: { p: params.noiseP },
    shots: params.shots,
    seed: params.seed,
    decoders: params.decoders
  };
}
