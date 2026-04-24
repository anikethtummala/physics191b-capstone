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
    distances: [...new Set(params.distances)].sort((a, b) => a - b),
    rounds: null,
    basis: params.basis,
    noise: { p: params.noiseP },
    shots: params.shots,
    seed: params.seed,
    decoders: params.decoders
  };
}
