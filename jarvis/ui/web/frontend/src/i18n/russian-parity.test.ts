import { expect, it } from "vitest";
import en from "./locales/en.json";
import ru from "./locales/ru.json";
import marketplaceEn from "./locales/marketplace/en.json";
import marketplaceRu from "./locales/marketplace/ru.json";
import modelsEn from "./locales/local_models/en.json";
import modelsRu from "./locales/local_models/ru.json";
import societyEn from "./locales/society/en.json";
import societyRu from "./locales/society/ru.json";

function flatten(value: unknown, prefix = ""): Record<string, string> {
  if (typeof value === "string") return { [prefix]: value };
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .flatMap(([key, child]) => Object.entries(flatten(child, prefix ? `${prefix}.${key}` : key))));
}

for (const [name, source, translation] of [
  ["core", en, ru],
  ["marketplace", marketplaceEn, marketplaceRu],
  ["local models", modelsEn, modelsRu],
  ["society", societyEn, societyRu],
] as const) {
  it(`Russian ${name} covers every English key and preserves interpolation tokens`, () => {
    const original = flatten(source);
    const translated = flatten(translation);
    expect(Object.keys(translated).sort()).toEqual(Object.keys(original).sort());
    for (const [key, text] of Object.entries(original)) {
      if (text.trim()) expect(translated[key]?.trim(), key).toBeTruthy();
      const tokens = (s: string) => (s.match(/\{[^{}]+\}/g) ?? []).sort();
      expect(tokens(translated[key]), key).toEqual(tokens(text));
    }
  });
}
