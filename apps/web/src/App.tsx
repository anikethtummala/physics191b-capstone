import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Database, Terminal } from "lucide-react";
import {
  fetchBenchmark,
  fetchBenchmarkSubmission,
  fetchHealth,
  startBenchmark
} from "./api";
import { buildBenchmarkRequest } from "./benchmark";
import { BenchmarkChart } from "./components/BenchmarkChart";
import { ControlRail } from "./components/ControlRail";
import { DecoderStrip } from "./components/DecoderStrip";
import { ReproducibilityPanel } from "./components/ReproducibilityPanel";
import { ResultsTable } from "./components/ResultsTable";
import type {
  Basis,
  BenchmarkJob,
  DecoderName,
  HealthResponse
} from "./types/api";

const DEFAULT_DECODERS: DecoderName[] = ["mwpm", "cnn", "gnn", "transformer"];

function App() {
  const [basis, setBasis] = useState<Basis>("x");
  const [distances, setDistances] = useState<number[]>([3, 5, 7]);
  const [noiseP, setNoiseP] = useState(0.001);
  const [shots, setShots] = useState(1000);
  const [seed, setSeed] = useState(1337);
  const [decoders, setDecoders] = useState<DecoderName[]>(DEFAULT_DECODERS);
  const [job, setJob] = useState<BenchmarkJob | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [submissionBusy, setSubmissionBusy] = useState(false);
  const [uiError, setUiError] = useState<string | null>(null);

  const activeRequest = useMemo(
    () =>
      buildBenchmarkRequest({
        distances,
        basis,
        noiseP,
        shots,
        seed,
        decoders
      }),
    [basis, decoders, distances, noiseP, seed, shots]
  );

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({ ok: false, quantum_stack: false, missing: ["api offline"] })
      );
  }, []);

  async function runBenchmark() {
    setBusy(true);
    setUiError(null);
    try {
      const jobId = await startBenchmark(activeRequest);
      setJob({
        job_id: jobId,
        status: "pending",
        progress: 0,
        results: [],
        errors: []
      });
      for (;;) {
        const next = await fetchBenchmark(jobId);
        setJob(next);
        if (next.status === "complete" || next.status === "error") {
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    } catch (error) {
      setUiError(error instanceof Error ? error.message : "Benchmark request failed.");
    } finally {
      setBusy(false);
      fetchHealth().then(setHealth).catch(() => undefined);
    }
  }

  async function downloadSubmission() {
    if (!job) return;
    setSubmissionBusy(true);
    setUiError(null);
    try {
      const submission = await fetchBenchmarkSubmission(job.job_id);
      const blob = new Blob([JSON.stringify(submission, null, 2)], {
        type: "application/json"
      });
      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${submission.suite_id}-${submission.suite_version}-${job.job_id}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      setUiError(error instanceof Error ? error.message : "Submission export failed.");
    } finally {
      setSubmissionBusy(false);
    }
  }

  function toggleDistance(distance: number) {
    setDistances((current) =>
      current.includes(distance)
        ? current.filter((value) => value !== distance)
        : [...current, distance].sort((a, b) => a - b)
    );
  }

  function toggleDecoder(decoder: DecoderName) {
    setDecoders((current) =>
      current.includes(decoder)
        ? current.filter((value) => value !== decoder)
        : [...current, decoder]
    );
  }

  return (
    <main className="app-shell">
      <ControlRail
        basis={basis}
        busy={busy}
        decoders={decoders}
        distances={distances}
        health={health}
        noiseP={noiseP}
        seed={seed}
        shots={shots}
        onBasisChange={setBasis}
        onDecoderToggle={toggleDecoder}
        onDistanceToggle={toggleDistance}
        onNoiseChange={setNoiseP}
        onRunBenchmark={runBenchmark}
        onSeedChange={setSeed}
        onShotsChange={setShots}
      />

      <section className="workspace">
        <motion.div
          className="topline"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div>
            <p className="eyebrow">Deterministic checks, probabilistic decoding</p>
            <h1>Decoder benchmark</h1>
          </div>
          <div className="status-row">
            <span className="status-pill">
              <Activity size={14} />
              {job ? `${Math.round(job.progress * 100)}%` : "idle"}
            </span>
            <span className="status-pill">
              <Database size={14} />
              {health?.quantum_stack ? "quantum stack ready" : "deps pending"}
            </span>
          </div>
        </motion.div>

        {uiError && (
          <div className="inline-error">
            <Terminal size={14} />
            <span>{uiError}</span>
          </div>
        )}

        <div className="benchmark-grid">
          <BenchmarkChart noiseP={noiseP} results={job?.results ?? []} shots={shots} />
          <DecoderStrip decoders={DEFAULT_DECODERS} results={job?.results ?? []} />
        </div>

        <div className="lower-grid">
          <ResultsTable results={job?.results ?? []} />
          <div className="right-stack">
            <ReproducibilityPanel
              busy={busy || submissionBusy}
              job={job}
              request={activeRequest}
              onDownloadSubmission={downloadSubmission}
            />
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
