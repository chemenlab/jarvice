import { afterEach, beforeEach, expect, it, vi } from "vitest";

beforeEach(() => {
  localStorage.clear();
  vi.resetModules();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.doUnmock("@/i18n/locales/ru.json");
});

it("hydrates Russian UI without changing the voice or reply language", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true,
    json: async () => ({ language: "ru" }),
  })));
  const i18n = await import("@/i18n");
  const before = i18n.useI18nStore.getState();
  await i18n.hydrateUiLanguage();
  expect(i18n.useI18nStore.getState().ui).toBe("ru");
  expect(i18n.useI18nStore.getState().reply).toBe(before.reply);
  expect(i18n.useI18nStore.getState().stt).toBe(before.stt);
  expect(i18n.translate("nav.settings")).toBe("Настройки");
});

it("restores Russian UI from local storage on a fresh mount", async () => {
  localStorage.setItem("jarvis.ui.language", "ru");
  const { useI18nStore } = await import("@/i18n");
  expect(useI18nStore.getState().ui).toBe("ru");
});

it("keeps a newer language choice when an older Russian hydration finishes loading", async () => {
  let releaseLocale!: () => void;
  const localeReady = new Promise<void>((resolve) => { releaseLocale = resolve; });
  let notifyLocaleStarted!: () => void;
  const localeStarted = new Promise<void>((resolve) => { notifyLocaleStarted = resolve; });
  vi.doMock("@/i18n/locales/ru.json", async (importOriginal) => {
    notifyLocaleStarted();
    await localeReady;
    return importOriginal();
  });
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true,
    json: async () => ({ language: "ru" }),
  })));

  const i18n = await import("@/i18n");
  const hydration = i18n.hydrateUiLanguage();
  await localeStarted;
  i18n.setUiLanguage("en");
  releaseLocale();
  await hydration;

  expect(i18n.useI18nStore.getState().ui).toBe("en");
  expect(localStorage.getItem("jarvis.ui.language")).toBe("en");
  expect(document.documentElement.lang).toBe("en");
  expect(i18n.translate("nav.settings")).toBe("Settings");
});
