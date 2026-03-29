/**
 * Text tokenizer for full-text search indexing and querying.
 *
 * Standard tokenizer: Unicode split → lowercase → ASCII fold → optional stemming → optional stopwords.
 * CJK tokenizer: bigram sliding window (no dictionary needed).
 */

import { stem } from "./stemmer.js";

// ASCII folding map for common diacriticals
const ASCII_FOLD: Record<string, string> = {};
// Explicit mappings — no clever parsing, just correct
const FOLD_MAP: [string, string][] = [
  ["à", "a"], ["á", "a"], ["â", "a"], ["ã", "a"], ["ä", "a"], ["å", "a"],
  ["è", "e"], ["é", "e"], ["ê", "e"], ["ë", "e"],
  ["ì", "i"], ["í", "i"], ["î", "i"], ["ï", "i"],
  ["ò", "o"], ["ó", "o"], ["ô", "o"], ["õ", "o"], ["ö", "o"], ["ø", "o"],
  ["ù", "u"], ["ú", "u"], ["û", "u"], ["ü", "u"],
  ["ý", "y"], ["ÿ", "y"],
  ["ñ", "n"], ["ç", "c"], ["ð", "d"],
  ["ß", "ss"], ["æ", "ae"], ["þ", "th"],
];
for (const [from, to] of FOLD_MAP) ASCII_FOLD[from] = to;

const ENGLISH_STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
  "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
  "such", "that", "the", "their", "then", "there", "these", "they",
  "this", "to", "was", "will", "with",
]);

/** CJK Unicode ranges */
function isCJK(code: number): boolean {
  return (
    (code >= 0x4E00 && code <= 0x9FFF) ||   // CJK Unified
    (code >= 0x3400 && code <= 0x4DBF) ||   // CJK Extension A
    (code >= 0x3040 && code <= 0x309F) ||   // Hiragana
    (code >= 0x30A0 && code <= 0x30FF) ||   // Katakana
    (code >= 0xAC00 && code <= 0xD7AF) ||   // Hangul
    (code >= 0xFF00 && code <= 0xFFEF)      // Fullwidth
  );
}

function isWordChar(code: number): boolean {
  // Letters, digits, underscore — fast ASCII path
  if (code >= 0x61 && code <= 0x7A) return true; // a-z
  if (code >= 0x41 && code <= 0x5A) return true; // A-Z
  if (code >= 0x30 && code <= 0x39) return true; // 0-9
  if (code === 0x5F) return true;                 // _
  // Unicode letters/digits (basic check — covers Latin-1 supplement + accented)
  if (code >= 0xC0 && code <= 0x24F) return true;
  return false;
}

export interface TokenizerConfig {
  stopwords?: "english" | "none" | Set<string>;
  /** Minimum token length to keep (default: 1) */
  minLength?: number;
  /** Enable Porter stemming for English (default: false) */
  stemming?: boolean;
}

export interface Token {
  /** The normalized token text */
  text: string;
  /** Character offset (UTF-16 code unit index) in the original string where this token starts */
  startOffset: number;
  /** Character offset (UTF-16 code unit index) in the original string where this token ends (exclusive) */
  endOffset: number;
}

/**
 * Tokenize text into normalized terms.
 * Handles mixed Latin + CJK text in a single pass.
 */
export function tokenize(text: string, config?: TokenizerConfig): Token[] {
  const tokens: Token[] = [];
  const stopwords = config?.stopwords === "none" ? undefined
    : config?.stopwords instanceof Set ? config.stopwords
    : ENGLISH_STOPWORDS;
  const minLen = config?.minLength ?? 1;

  let i = 0;
  const len = text.length;

  while (i < len) {
    const code = text.charCodeAt(i);

    // CJK: emit bigrams
    if (isCJK(code)) {
      // Collect consecutive CJK characters
      const cjkChars: string[] = [];
      const cjkOffsets: number[] = [];
      while (i < len && isCJK(text.charCodeAt(i))) {
        cjkOffsets.push(i);
        cjkChars.push(text[i]);
        i++;
      }
      // Emit bigrams
      for (let j = 0; j < cjkChars.length - 1; j++) {
        tokens.push({
          text: cjkChars[j] + cjkChars[j + 1],
          startOffset: cjkOffsets[j],
          endOffset: cjkOffsets[j + 1] + 1,
        });
      }
      // Single CJK char at end (or alone) — emit as unigram
      if (cjkChars.length === 1) {
        tokens.push({
          text: cjkChars[0],
          startOffset: cjkOffsets[0],
          endOffset: cjkOffsets[0] + 1,
        });
      }
      continue;
    }

    // Latin/number word
    if (isWordChar(code)) {
      const start = i;
      let word = "";
      while (i < len) {
        const c = text.charCodeAt(i);
        if (!isWordChar(c) && !isCJK(c)) break;
        if (isCJK(c)) break; // switch to CJK mode
        // Lowercase + ASCII fold in one pass
        let ch = String.fromCharCode(c).toLowerCase();
        if (ASCII_FOLD[ch]) ch = ASCII_FOLD[ch];
        word += ch;
        i++;
      }
      if (word.length >= minLen && (!stopwords || !stopwords.has(word))) {
        const final = config?.stemming ? stem(word) : word;
        tokens.push({ text: final, startOffset: start, endOffset: i });
      }
      continue;
    }

    // Skip non-word characters (whitespace, punctuation, emoji)
    i++;
  }

  return tokens;
}

/**
 * Tokenize and return just the term strings (no offsets).
 * Faster for indexing where positions aren't needed.
 */
export function tokenizeTerms(text: string, config?: TokenizerConfig): string[] {
  return tokenize(text, config).map(t => t.text);
}

/**
 * Check if text is primarily CJK (for auto-detecting tokenizer mode).
 */
export function isPrimarilyCJK(text: string): boolean {
  let cjk = 0;
  let total = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code > 0x20) { // skip whitespace
      total++;
      if (isCJK(code)) cjk++;
    }
  }
  return total > 0 && cjk / total > 0.5;
}
