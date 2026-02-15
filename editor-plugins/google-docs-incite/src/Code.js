/**
 * inCite for Google Docs — server-side Apps Script.
 *
 * Handles: menu creation, sidebar display, context extraction from
 * the document cursor, citation insertion, and settings persistence.
 *
 * Known issue: google.script.run fails with PERMISSION_DENIED / warden
 * TransportError when multiple Google accounts are signed in. The sidebar's
 * internal bridge (warden) can't resolve the correct account context.
 * Workaround: use Incognito with a single account, or a dedicated Chrome
 * profile. TODO: investigate deploying as a published add-on with explicit
 * account binding, or detect the multi-account condition in the sidebar JS
 * and show a helpful message.
 */

// ── Diagnostics ─────────────────────────────────────────────────

/** Run from script editor or sidebar to test permissions step by step. */
function testAccess() {
  var results = [];

  try {
    var props = PropertiesService.getUserProperties().getProperties();
    results.push('PropertiesService: OK');
  } catch (e) {
    results.push('PropertiesService: FAIL - ' + e.message);
  }

  try {
    var doc = DocumentApp.getActiveDocument();
    results.push('getActiveDocument: OK - ' + doc.getName());
  } catch (e) {
    results.push('getActiveDocument: FAIL - ' + e.message);
  }

  try {
    var doc = DocumentApp.getActiveDocument();
    var cursor = doc.getCursor();
    results.push('getCursor: OK - ' + (cursor ? 'has cursor' : 'no cursor'));
  } catch (e) {
    results.push('getCursor: FAIL - ' + e.message);
  }

  try {
    var doc = DocumentApp.getActiveDocument();
    var body = doc.getBody();
    var text = body.getText().substring(0, 100);
    results.push('getBody/getText: OK - "' + text + '"');
  } catch (e) {
    results.push('getBody/getText: FAIL - ' + e.message);
  }

  Logger.log(results.join('\n'));
  return results;
}

// ── Menu & sidebar ──────────────────────────────────────────────

function onOpen() {
  DocumentApp.getUi()
    .createMenu('inCite')
    .addItem('Open Recommendations Panel', 'showSidebar')
    .addItem('Settings…', 'showSettings')
    .addToUi();
}

function showSidebar() {
  var html = HtmlService.createTemplateFromFile('src/Sidebar')
    .evaluate()
    .setTitle('inCite')
    .setWidth(360);
  DocumentApp.getUi().showSidebar(html);
}

function showSettings() {
  var html = HtmlService.createTemplateFromFile('src/Settings')
    .evaluate()
    .setWidth(420)
    .setHeight(480);
  DocumentApp.getUi().showModalDialog(html, 'inCite Settings');
}

/** Include helper for HTML scriptlets: <?!= include('src/Sidebar.css') ?> */
function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename).getContent();
}

// ── Settings (UserProperties) ───────────────────────────────────

var DEFAULTS = {
  apiUrl: 'http://127.0.0.1:8230',
  k: 10,
  authorBoost: 1.0,
  contextSentences: 6,
  insertFormat: '({first_author}, {year})',
  showParagraphs: true
};

function getSettings() {
  var props = PropertiesService.getUserProperties().getProperties();
  return {
    apiUrl:            props.apiUrl            || DEFAULTS.apiUrl,
    k:                 parseInt(props.k, 10)   || DEFAULTS.k,
    authorBoost:       parseFloat(props.authorBoost) || DEFAULTS.authorBoost,
    contextSentences:  parseInt(props.contextSentences, 10) || DEFAULTS.contextSentences,
    insertFormat:      props.insertFormat      || DEFAULTS.insertFormat,
    showParagraphs:    props.showParagraphs !== undefined ? props.showParagraphs === 'true' : DEFAULTS.showParagraphs
  };
}

function saveSettings(settings) {
  var store = {};
  for (var key in settings) {
    store[key] = String(settings[key]);
  }
  PropertiesService.getUserProperties().setProperties(store);
}

// ── Context extraction ──────────────────────────────────────────

/**
 * Extract text around the cursor for use as a recommendation query.
 *
 * Google Docs cursor gives us the element (paragraph) the cursor is in.
 * We walk neighboring paragraphs, split into sentences, and return a
 * window centered on the cursor paragraph — analogous to the Obsidian
 * context extractor but paragraph-based instead of character-offset-based.
 *
 * @return {Object} {text, cursorSentenceIndex, totalSentences} or {error}
 */
function getContextAtCursor() {
  var doc = DocumentApp.getActiveDocument();
  var cursor = doc.getCursor();
  var selection = doc.getSelection();
  var body = doc.getBody();
  var numChildren = body.getNumChildren();

  // If user has selected text, use that directly
  if (selection) {
    var elements = selection.getRangeElements();
    var parts = [];
    for (var i = 0; i < elements.length; i++) {
      var el = elements[i].getElement();
      if (el.asText) {
        var text = el.asText().getText();
        if (elements[i].isPartial()) {
          text = text.substring(
            elements[i].getStartOffset(),
            elements[i].getEndOffsetInclusive() + 1
          );
        }
        if (text.trim()) parts.push(text.trim());
      }
    }
    if (parts.length > 0) {
      // Save insertion point at end of selection
      var lastRange = elements[elements.length - 1];
      var lastEl = lastRange.getElement();
      var endOffset = lastRange.isPartial()
        ? lastRange.getEndOffsetInclusive() + 1
        : (lastEl.asText ? lastEl.asText().getText().length : 0);
      // Find paragraph index for the last selected element
      var selParent = lastEl;
      while (selParent.getParent() !== body && selParent.getParent() !== null) {
        selParent = selParent.getParent();
      }
      for (var pi = 0; pi < numChildren; pi++) {
        if (body.getChild(pi) === selParent) {
          var selCache = CacheService.getUserCache();
          selCache.put('mc_paraIndex', String(pi), 600);
          selCache.put('mc_charOffset', String(endOffset), 600);
          break;
        }
      }

      var selectedText = stripCitations(parts.join(' '));
      return { text: selectedText, cursorSentenceIndex: 0, totalSentences: 1 };
    }
  }

  // No selection — extract context around cursor
  if (!cursor) {
    return { error: 'Place your cursor in the document text first.' };
  }

  var cursorElement = cursor.getElement();

  // Find the paragraph index containing the cursor
  var cursorParagraphIndex = -1;
  for (var i = 0; i < numChildren; i++) {
    var child = body.getChild(i);
    if (child.asText && child.asText() === cursorElement.asText()) {
      cursorParagraphIndex = i;
      break;
    }
    // Also check if cursor is inside a child of this element
    if (child === cursorElement || child === cursorElement.getParent()) {
      cursorParagraphIndex = i;
      break;
    }
  }

  if (cursorParagraphIndex === -1) {
    // Fallback: walk all paragraphs and find one containing the cursor text
    var cursorText = '';
    if (cursorElement.asText) {
      cursorText = cursorElement.asText().getText();
    }
    for (var i = 0; i < numChildren; i++) {
      var child = body.getChild(i);
      if (child.getType() === DocumentApp.ElementType.PARAGRAPH) {
        if (child.asText().getText() === cursorText) {
          cursorParagraphIndex = i;
          break;
        }
      }
    }
  }

  if (cursorParagraphIndex === -1) {
    return { error: 'Could not determine cursor position. Click inside a paragraph.' };
  }

  // Save cursor position for insertCitation (sidebar click steals doc focus)
  saveCursorPosition_(cursor, cursorParagraphIndex);

  var settings = getSettings();
  var windowSentences = settings.contextSentences;

  // Collect text from surrounding paragraphs
  var paragraphWindow = Math.ceil(windowSentences / 2) + 2; // extra paragraphs to ensure enough sentences
  var startIdx = Math.max(0, cursorParagraphIndex - paragraphWindow);
  var endIdx = Math.min(numChildren, cursorParagraphIndex + paragraphWindow + 1);

  var beforeText = [];
  var cursorParaText = '';
  var afterText = [];

  for (var i = startIdx; i < endIdx; i++) {
    var child = body.getChild(i);
    var text = '';
    if (child.asText) {
      text = child.asText().getText().trim();
    }
    if (!text) continue;

    if (i < cursorParagraphIndex) {
      beforeText.push(text);
    } else if (i === cursorParagraphIndex) {
      cursorParaText = text;
    } else {
      afterText.push(text);
    }
  }

  // Split all text into sentences
  var allText = beforeText.concat([cursorParaText], afterText).join(' ');
  var sentences = splitSentences(allText);

  if (sentences.length === 0) {
    return { error: 'No text found around cursor.' };
  }

  // Find which sentence is closest to the cursor paragraph text
  var cursorSentence = 0;
  var cursorParaStart = allText.indexOf(cursorParaText);
  // Find first sentence that overlaps with cursor paragraph
  var accum = 0;
  for (var i = 0; i < sentences.length; i++) {
    var idx = allText.indexOf(sentences[i], accum);
    if (idx === -1) continue;
    var sentEnd = idx + sentences[i].length;
    if (idx <= cursorParaStart && sentEnd > cursorParaStart) {
      cursorSentence = i;
      break;
    }
    if (idx > cursorParaStart) {
      cursorSentence = Math.max(0, i - 1);
      break;
    }
    accum = sentEnd;
    cursorSentence = i;
  }

  // Extract window centered on cursor sentence
  var halfWindow = Math.floor(windowSentences / 2);
  var start = Math.max(0, cursorSentence - halfWindow);
  var end = Math.min(sentences.length, cursorSentence + halfWindow + 1);

  var selected = sentences.slice(start, end);
  var rawText = selected.join(' ');
  var cleanText = stripCitations(rawText);

  return {
    text: cleanText,
    cursorSentenceIndex: cursorSentence - start,
    totalSentences: selected.length
  };
}

// ── Cursor position cache ────────────────────────────────────────

/**
 * Save cursor position to cache so insertCitation can use it later.
 * Clicking sidebar buttons steals focus from the doc, so getCursor()
 * returns a degraded position by the time insertCitation runs.
 * We capture the precise position here, during getContextAtCursor(),
 * when the cursor is still valid.
 */
function saveCursorPosition_(cursor, paragraphIndex) {
  var element = cursor.getElement();
  var offset = cursor.getOffset();

  // Compute character offset within the paragraph's text.
  // For TEXT elements, offset is already the character position.
  // For PARAGRAPH/container elements, offset is a child-element index.
  var charOffset = 0;
  if (element.getType() === DocumentApp.ElementType.TEXT) {
    // If there are sibling Text runs before this one, sum their lengths
    var parent = element.getParent();
    if (parent && parent.getNumChildren) {
      for (var i = 0; i < parent.getNumChildren(); i++) {
        var child = parent.getChild(i);
        if (child.asText && child.asText().getText() === element.asText().getText()
            && child.getType() === element.getType()) {
          break;
        }
        if (child.asText) {
          charOffset += child.asText().getText().length;
        }
      }
    }
    charOffset += offset;
  } else if (element.getNumChildren) {
    // Container: offset = child index; sum text of children before cursor
    for (var i = 0; i < offset && i < element.getNumChildren(); i++) {
      var child = element.getChild(i);
      if (child.asText) {
        charOffset += child.asText().getText().length;
      }
    }
  }

  var cache = CacheService.getUserCache();
  cache.put('mc_paraIndex', String(paragraphIndex), 600);
  cache.put('mc_charOffset', String(charOffset), 600);
}

// ── Citation insertion ──────────────────────────────────────────

/**
 * Insert a formatted citation string at the saved cursor position.
 *
 * Uses the position cached by getContextAtCursor() rather than the
 * live cursor, because clicking in the sidebar steals doc focus and
 * getCursor() returns a degraded/wrong position afterward.
 *
 * @param {string} citationText  The pre-formatted citation string.
 */
function insertCitation(citationText) {
  var doc = DocumentApp.getActiveDocument();
  var body = doc.getBody();

  // Use saved position from the last getContextAtCursor() call
  var cache = CacheService.getUserCache();
  var savedParaIndex = cache.get('mc_paraIndex');
  var savedCharOffset = cache.get('mc_charOffset');

  if (savedParaIndex !== null && savedCharOffset !== null) {
    var paraIndex = parseInt(savedParaIndex, 10);
    var charOffset = parseInt(savedCharOffset, 10);

    if (paraIndex >= 0 && paraIndex < body.getNumChildren()) {
      var para = body.getChild(paraIndex);
      var text = para.editAsText();
      var textLength = text.getText().length;

      if (textLength === 0) {
        text.setText(citationText);
      } else {
        var insertPos = Math.min(charOffset, textLength);
        text.insertText(insertPos, citationText);
        cache.put('mc_charOffset', String(insertPos + citationText.length), 600);
      }
      return;
    }
  }

  // No saved position — try live cursor as fallback
  var cursor = doc.getCursor();
  if (cursor) {
    var element = cursor.getElement();
    var offset = cursor.getOffset();
    if (element.getType() === DocumentApp.ElementType.TEXT) {
      element.asText().insertText(offset, citationText);
      return;
    }
  }

  DocumentApp.getUi().alert(
    'Could not determine cursor position. ' +
    'Click in your document and run Get Recommendations before inserting.'
  );
}

// ── Text utilities (ported from Obsidian context-extractor.ts) ──

/**
 * Split text into sentences using regex heuristics.
 * Handles common abbreviations to avoid false splits.
 */
function splitSentences(text) {
  // Abbreviation pattern — don't split after these
  var abbrevPattern = /(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|Fig|Figs|Eq|Eqs|al|vs|i\.e|e\.g|cf|no|vol|pp|ed|Rev|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s*/gi;

  var placeholder = '\x00';
  var processed = text.replace(abbrevPattern, function(match) {
    return match.replace(/\.\s*/, placeholder);
  });

  // Split on sentence-ending punctuation followed by whitespace and uppercase
  var parts = processed.split(/(?<=[.!?])\s+(?=[A-Z"])/);

  return parts
    .map(function(s) {
      return s.replace(new RegExp(placeholder, 'g'), '. ').trim();
    })
    .filter(function(s) { return s.length > 0; });
}

/**
 * Strip citation markers from text before sending to the API.
 * Handles: [@key], [cite], \cite{key}, (Author, YYYY), and Google Docs
 * superscript-style [1] or [1,2,3] numeric references.
 */
function stripCitations(text) {
  return text
    .replace(/\[@[^\]]*\]/g, '')          // [@key]
    .replace(/\[cite\]/gi, '')             // [cite]
    .replace(/\\cite\{[^}]*\}/g, '')       // \cite{key}
    .replace(/\[\d+(?:,\s*\d+)*\]/g, '')   // [1] or [1,2,3]
    .replace(/\s{2,}/g, ' ')              // collapse multiple spaces
    .trim();
}
