/**
 * Porter stemmer for English.
 *
 * Reduces words to their root form: "running" → "run", "cats" → "cat".
 * Implementation follows Martin Porter's original algorithm (1980).
 * Reference: https://tartarus.org/martin/PorterStemmer/def.txt
 */

function isConsonant(word: string, i: number): boolean {
  const c = word[i];
  if (c === "a" || c === "e" || c === "i" || c === "o" || c === "u") return false;
  if (c === "y") return i === 0 || !isConsonant(word, i - 1);
  return true;
}

/** Count consonant-vowel sequences (measure) in word[0..k]. */
function measure(word: string, k: number): number {
  let n = 0;
  let i = 0;
  while (i <= k && isConsonant(word, i)) i++;
  if (i > k) return 0;
  while (i <= k) {
    while (i <= k && !isConsonant(word, i)) i++;
    if (i > k) break;
    n++;
    while (i <= k && isConsonant(word, i)) i++;
  }
  return n;
}

function endsWith(word: string, suffix: string): boolean {
  return word.endsWith(suffix);
}

function hasVowel(word: string, k: number): boolean {
  for (let i = 0; i <= k; i++) {
    if (!isConsonant(word, i)) return true;
  }
  return false;
}

function endsWithDoubleConsonant(word: string): boolean {
  const len = word.length;
  if (len < 2) return false;
  return word[len - 1] === word[len - 2] && isConsonant(word, len - 1);
}

function endsWithCVC(word: string): boolean {
  const len = word.length;
  if (len < 3) return false;
  const c = word[len - 1];
  return isConsonant(word, len - 1) && !isConsonant(word, len - 2) && isConsonant(word, len - 3) &&
    c !== "w" && c !== "x" && c !== "y";
}

function replaceSuffix(word: string, suffix: string, replacement: string): string {
  return word.slice(0, -suffix.length) + replacement;
}

/**
 * Apply the Porter stemming algorithm to a single word.
 * Returns the stemmed form.
 */
export function stem(word: string): string {
  if (word.length <= 2) return word;

  // Step 1a
  if (endsWith(word, "sses")) word = replaceSuffix(word, "sses", "ss");
  else if (endsWith(word, "ies")) word = replaceSuffix(word, "ies", "i");
  else if (!endsWith(word, "ss") && endsWith(word, "s")) word = word.slice(0, -1);

  // Step 1b
  let step1bFlag = false;
  if (endsWith(word, "eed")) {
    const stem = word.slice(0, -3);
    if (measure(stem, stem.length - 1) > 0) word = replaceSuffix(word, "eed", "ee");
  } else if (endsWith(word, "ed")) {
    const stem = word.slice(0, -2);
    if (hasVowel(stem, stem.length - 1)) {
      word = stem;
      step1bFlag = true;
    }
  } else if (endsWith(word, "ing")) {
    const stem = word.slice(0, -3);
    if (hasVowel(stem, stem.length - 1)) {
      word = stem;
      step1bFlag = true;
    }
  }

  if (step1bFlag) {
    if (endsWith(word, "at")) word += "e";
    else if (endsWith(word, "bl")) word += "e";
    else if (endsWith(word, "iz")) word += "e";
    else if (endsWithDoubleConsonant(word) && !endsWith(word, "l") && !endsWith(word, "s") && !endsWith(word, "z")) {
      word = word.slice(0, -1);
    } else if (measure(word, word.length - 1) === 1 && endsWithCVC(word)) {
      word += "e";
    }
  }

  // Step 1c
  if (endsWith(word, "y") && hasVowel(word, word.length - 2)) {
    word = word.slice(0, -1) + "i";
  }

  // Step 2
  const step2: [string, string][] = [
    ["ational", "ate"], ["tional", "tion"], ["enci", "ence"], ["anci", "ance"],
    ["izer", "ize"], ["abli", "able"], ["alli", "al"], ["entli", "ent"],
    ["eli", "e"], ["ousli", "ous"], ["ization", "ize"], ["ation", "ate"],
    ["ator", "ate"], ["alism", "al"], ["iveness", "ive"], ["fulness", "ful"],
    ["ousness", "ous"], ["aliti", "al"], ["iviti", "ive"], ["biliti", "ble"],
  ];
  for (const [suffix, replacement] of step2) {
    if (endsWith(word, suffix)) {
      const stem = word.slice(0, -suffix.length);
      if (measure(stem, stem.length - 1) > 0) word = stem + replacement;
      break;
    }
  }

  // Step 3
  const step3: [string, string][] = [
    ["icate", "ic"], ["ative", ""], ["alize", "al"], ["iciti", "ic"],
    ["ical", "ic"], ["ful", ""], ["ness", ""],
  ];
  for (const [suffix, replacement] of step3) {
    if (endsWith(word, suffix)) {
      const stem = word.slice(0, -suffix.length);
      if (measure(stem, stem.length - 1) > 0) word = stem + replacement;
      break;
    }
  }

  // Step 4
  const step4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
    "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
    "ous", "ive", "ize",
  ];
  for (const suffix of step4) {
    if (endsWith(word, suffix)) {
      const stem = word.slice(0, -suffix.length);
      if (suffix === "ion") {
        if (stem.length > 0 && (stem.endsWith("s") || stem.endsWith("t")) && measure(stem, stem.length - 1) > 1) {
          word = stem;
        }
      } else if (measure(stem, stem.length - 1) > 1) {
        word = stem;
      }
      break;
    }
  }

  // Step 5a
  if (endsWith(word, "e")) {
    const stem = word.slice(0, -1);
    const m = measure(stem, stem.length - 1);
    if (m > 1 || (m === 1 && !endsWithCVC(stem))) word = stem;
  }

  // Step 5b
  if (measure(word, word.length - 1) > 1 && endsWithDoubleConsonant(word) && endsWith(word, "l")) {
    word = word.slice(0, -1);
  }

  return word;
}
