import { Download, ShieldAlert, ShieldCheck } from "lucide-react";
import type { BenchmarkJob, BenchmarkRequest } from "../types/api";

interface ReproducibilityPanelProps {
  busy: boolean;
  job: BenchmarkJob | null;
  request: BenchmarkRequest;
  onDownloadSubmission: () => void;
}

export function ReproducibilityPanel({
  busy,
  job,
  request,
  onDownloadSubmission
}: ReproducibilityPanelProps) {
  const latestComplete = [...(job?.results ?? [])]
    .reverse()
    .find((result) => result.status === "complete");
  const environment = job?.runtime_environment;
  const suiteCompliant = job?.suite_compliant ?? false;
  const complianceErrors = job?.suite_compliance_errors ?? [];

  return (
    <section className="panel compact-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Public benchmark</p>
          <h2>Reproducibility</h2>
        </div>
        {suiteCompliant ? <ShieldCheck size={18} /> : <ShieldAlert size={18} />}
      </div>
      <div className="repro-grid">
        <span>Suite</span>
        <strong>
          {job?.suite_id ?? request.suite_id} / {job?.suite_version ?? request.suite_version}
        </strong>
        <span>Seed</span>
        <strong>{request.seed ?? "unfixed"}</strong>
        <span>Confidence</span>
        <strong>{formatConfidence(latestComplete)}</strong>
        <span>Runtime</span>
        <strong>{formatEnvironment(environment)}</strong>
        <span>Trace</span>
        <strong>{formatShortHash(latestComplete?.trace_id)}</strong>
      </div>
      {complianceErrors.length > 0 && (
        <p className="quiet-text">{complianceErrors.join(" ")}</p>
      )}
      <button
        className="secondary-action full-width"
        disabled={!job || busy}
        type="button"
        onClick={onDownloadSubmission}
      >
        <Download size={16} />
        Submission JSON
      </button>
    </section>
  );
}

function formatConfidence(
  result: BenchmarkJob["results"][number] | undefined
): string {
  if (!result?.confidence_method || !result.confidence_level) return "pending";
  return `${Math.round(result.confidence_level * 100)}% ${result.confidence_method}`;
}

function formatEnvironment(environment: BenchmarkJob["runtime_environment"]): string {
  if (!environment) return "pending";
  return `${environment.platform} ${environment.machine} / Python ${environment.python}`;
}

function formatShortHash(hash: string | null | undefined): string {
  if (!hash) return "pending";
  return hash.slice(0, 10);
}
