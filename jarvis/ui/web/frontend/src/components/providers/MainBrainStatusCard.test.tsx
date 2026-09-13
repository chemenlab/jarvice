import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MainBrainStatusCard } from "./MainBrainStatusCard";

const status = {
  active_provider: "gemini", configured_provider: "gemini", requires_restart: false,
  codex: { ready: true, auth_mode: "chatgpt", transport: "app-server", model: "gpt-5.5" },
  voice: { mode: "pipeline", stt_provider: "gemini-api", tts_provider: "gemini-flash-tts",
    realtime_provider: "gemini-live", speech_key_available: true },
};
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn(async () => ({ ok: true, json: async () => structuredClone(status) }));
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("shows actual brain and Gemini speech independently without labelling pipeline Live", async () => {
  render(<MainBrainStatusCard />);
  await waitFor(() => expect(screen.getByTestId("main-brain-active").textContent).toBe("Gemini"));
  expect(screen.getByTestId("main-brain-voice").textContent).toMatch(/Gemini.*recognition and speech/i);
  expect(screen.getByTestId("main-brain-voice").textContent).not.toMatch(/Live/);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: /refresh/i }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
});

it("persists the choice and shows pending restart while retaining the actual brain", async () => {
  fetchMock.mockImplementation(async (_url, init) => ({ ok: true,
    json: async () => init?.method === "PUT"
      ? { ok: true, new_provider: "codex-subscription", applied_live: false, requires_restart: true }
      : structuredClone(status),
  }));
  render(<MainBrainStatusCard />);
  fireEvent.click(await screen.findByRole("button", { name: /use Codex as main brain/i }));
  await waitFor(() => expect(screen.getByTestId("main-brain-pending").textContent).toMatch(/restart/i));
  expect(screen.getByTestId("main-brain-active").textContent).toBe("Gemini");
  expect(fetchMock.mock.calls[1][0]).toBe("/api/settings/main-brain");
  expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ provider: "codex-subscription" });
  expect(fetchMock).toHaveBeenCalledTimes(2);
});

it("names Gemini Live only when the active voice mode is realtime", async () => {
  fetchMock.mockResolvedValue({ ok: true, json: async () => ({ ...status,
    voice: { ...status.voice, mode: "realtime" },
  }) });
  render(<MainBrainStatusCard />);
  await waitFor(() => expect(screen.getByTestId("main-brain-voice").textContent).toBe("Gemini Live"));
  expect(screen.getByText(/answers independently/i)).toBeTruthy();
});

it("shows a compatible live switch immediately without a restart notice", async () => {
  fetchMock.mockImplementation(async (_url, init) => ({ ok: true,
    json: async () => init?.method === "PUT"
      ? { ok: true, new_provider: "codex-subscription", applied_live: true, requires_restart: false }
      : structuredClone(status),
  }));
  render(<MainBrainStatusCard />);
  fireEvent.click(await screen.findByRole("button", { name: /use Codex as main brain/i }));
  await waitFor(() => expect(screen.getByTestId("main-brain-active").textContent).toBe("Codex CLI · ChatGPT subscription"));
  expect(screen.queryByTestId("main-brain-pending")).toBeNull();
});

it("does not let an API-key Codex login masquerade as a subscription", async () => {
  fetchMock.mockResolvedValue({ ok: true, json: async () => ({ ...status,
    codex: { ...status.codex, auth_mode: "apikey" },
  }) });
  render(<MainBrainStatusCard />);
  const button = await screen.findByRole("button", { name: /use Codex as main brain/i });
  expect((button as HTMLButtonElement).disabled).toBe(true);
});

it("renders a fast status with nullable auth while an explicit probe is pending", async () => {
  fetchMock.mockImplementation(async (url) => {
    if (String(url).endsWith("/probe")) return await new Promise(() => {});
    return { ok: true, json: async () => ({ ...status, codex: null,
      voice: { ...status.voice, speech_key_available: null } }) };
  });
  render(<MainBrainStatusCard />);
  await waitFor(() => expect(screen.getByTestId("main-brain-active").textContent).toBe("Gemini"));
  expect(screen.queryByText(/load.*failed/i)).toBeNull();
});
