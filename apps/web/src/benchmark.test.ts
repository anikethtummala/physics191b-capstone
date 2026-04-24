import { describe, expect, it } from "vitest";
import { buildBenchmarkRequest } from "./benchmark";

describe("buildBenchmarkRequest", () => {
  it("deduplicates and sorts distances while preserving benchmark controls", () => {
    const request = buildBenchmarkRequest({
      distances: [7, 3, 5, 3],
      basis: "z",
      noiseP: 0.002,
      shots: 256,
      seed: 22,
      decoders: ["transformer", "mwpm"]
    });

    expect(request.distances).toEqual([3, 5, 7]);
    expect(request.rounds).toBeNull();
    expect(request.noise.p).toBe(0.002);
    expect(request.decoders).toEqual(["transformer", "mwpm"]);
  });
});
