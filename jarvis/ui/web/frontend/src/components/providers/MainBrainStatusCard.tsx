import { useCallback, useEffect, useRef, useState } from "react";
import { Brain, Mic, RefreshCw } from "lucide-react";
import { useT } from "@/i18n";
import { prettyProviderName } from "@/lib/prettyProviderName";
import { BrainModelSelector } from "@/components/BrainModelSelector";

interface MainBrainStatus {
  active_provider: string;
  active_model?: string;
  configured_provider: string;
  requires_restart: boolean;
  codex: { ready: boolean; auth_mode: string; transport?: string; model?: string; error?: string } | null;
  voice: {
    mode: string;
    stt_provider: string;
    tts_provider: string;
    realtime_provider?: string | null;
    speech_key_available: boolean | null;
  };
}

const ENDPOINT = "/api/settings/main-brain";

/** Explicit snapshots: probing the CLI is a user action, never a polling loop. */
export function MainBrainStatusCard() {
  const t = useT();
  const [status, setStatus] = useState<MainBrainStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const pendingRequest = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    pendingRequest.current?.abort();
    const controller = new AbortController();
    pendingRequest.current = controller;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(ENDPOINT, { signal: controller.signal, cache: "no-store" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `HTTP ${response.status}`);
      if (!body.voice || typeof body.active_provider !== "string") {
        throw new Error();
      }
      if (!controller.signal.aborted) setStatus(body);
      if (!body.codex) {
        void fetch(`${ENDPOINT}/probe`, { method: "POST", signal: controller.signal })
          .then(async (probeResponse) => {
            const probe = await probeResponse.json();
            if (!probeResponse.ok) throw new Error(probe.detail ?? `HTTP ${probeResponse.status}`);
            if (!controller.signal.aborted) setStatus((current) => current && ({
              ...current, codex: probe,
              voice: { ...current.voice, speech_key_available: probe.speech_key_available },
            }));
          })
          .catch((cause) => {
            if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "");
          });
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        setStatus(null);
        setError(cause instanceof Error ? cause.message : "");
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const configured = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      if (!detail?.new_provider) return;
      setStatus((current) => current && ({
        ...current,
        configured_provider: detail.new_provider,
        active_provider: detail.applied_live ? detail.new_provider : current.active_provider,
        requires_restart: detail.requires_restart === true,
      }));
    };
    window.addEventListener("jarvis:main-brain-configured", configured);
    return () => {
      pendingRequest.current?.abort();
      window.removeEventListener("jarvis:main-brain-configured", configured);
    };
  }, [refresh]);

  async function selectCodex() {
    setSaving(true);
    setError("");
    try {
      const response = await fetch(ENDPOINT, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: "codex-subscription" }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? `HTTP ${response.status}`);
      window.dispatchEvent(new CustomEvent("jarvis:main-brain-configured", { detail: result }));
      if (result.applied_live) {
        window.dispatchEvent(new CustomEvent("jarvis:brain-switched"));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "");
    } finally {
      setSaving(false);
    }
  }

  const subscriptionReady = status?.codex?.ready && status.codex.auth_mode === "chatgpt";
  const activeCodex = status?.active_provider === "codex-subscription";
  const pipeline = status?.voice.mode === "pipeline";
  const geminiSpeech = pipeline && status.voice.stt_provider === "gemini-api"
    && status.voice.tts_provider === "gemini-flash-tts";
  const voiceLabel = !status ? "—" : pipeline
    ? geminiSpeech ? t("main_brain.gemini_speech")
      : `${prettyProviderName(status.voice.stt_provider)} → ${prettyProviderName(status.voice.tts_provider)}`
    : prettyProviderName(status.voice.realtime_provider ?? "");

  return (
    <section aria-label={t("main_brain.title")} className="mx-6 my-3 shrink-0 rounded-lg border border-border bg-card px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div className="flex min-w-0 items-center gap-2">
          <Brain className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
          <span className="text-muted-foreground">{t("main_brain.brain")}</span>
          <strong data-testid="main-brain-active" className="font-medium">
            {status ? activeCodex ? t("main_brain.codex_subscription") : prettyProviderName(status.active_provider) : "—"}
          </strong>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          <Mic className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
          <span className="text-muted-foreground">{t("main_brain.voice")}</span>
          <span data-testid="main-brain-voice">{voiceLabel}</span>
        </div>
        <button type="button" onClick={() => void refresh()} disabled={loading || saving}
          title={t("main_brain.refresh")} aria-label={t("main_brain.refresh")}
          className="ml-auto rounded p-1.5 text-muted-foreground hover:bg-secondary disabled:opacity-50">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} aria-hidden />
        </button>
      </div>
      {status && (
        <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          <span>{!status.codex ? t("main_brain.checking") : subscriptionReady ? t("main_brain.ready") : t("main_brain.login_needed")}</span>
          {status.voice.speech_key_available === false && <span>{t("main_brain.speech_key_needed")}</span>}
          {!pipeline && <span>{t("main_brain.live_independent")}</span>}
          {status.requires_restart ? (
            <span data-testid="main-brain-pending" role="status" className="text-primary">
              {t("main_brain.restart_required")}
            </span>
          ) : (
            <button type="button" onClick={() => void selectCodex()}
              disabled={loading || saving || !subscriptionReady || !status.voice.speech_key_available || (activeCodex && geminiSpeech)}
              className="ml-auto rounded-md border border-border px-3 py-1.5 text-foreground hover:bg-secondary disabled:opacity-50">
              {saving ? t("main_brain.saving") : t("main_brain.select_codex")}
            </button>
          )}
        </div>
      )}
      {activeCodex && (
        <div className="mt-3 border-t border-border pt-3">
          <BrainModelSelector providerId="codex-subscription" currentModel={status?.active_model}
            recommendedModel="gpt-5.6-sol" healthSection="brain" healthActive />
        </div>
      )}
      {!status && !loading && <p role="status" className="mt-2 text-xs text-muted-foreground">{t("main_brain.load_failed")}</p>}
      {error && <p role="alert" className="mt-2 text-xs text-destructive">{error}</p>}
    </section>
  );
}
