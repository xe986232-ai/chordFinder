// complex_chord.js
// Port JS dari complex_chord.py — HANYA bagian yang dipakai oleh XHMMDecoder
// (parsing template chord list + shifting antar 12 nada dasar).
// Bukan port lengkap (parse_chord_type/decode dari mir dataset builder yang
// tidak dipakai saat inference tidak diikutkan).

(function (global) {
  'use strict';

  const NUM_TO_ABS_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

  const TriadTypes = { x: -2, none: 0, maj: 1, min: 2, sus4: 3, sus2: 4, dim: 5, aug: 6, power: 7, one: 8 };
  const SeventhTypes = { unknown: -2, not_care: -1, none: 0, add_7: 1, add_b7: 2, add_bb7: 3 };
  const NinthTypes = { unknown: -2, not_care: -1, none: 0, add_9: 1, add_s9: 2, add_b9: 3 };
  const EleventhTypes = { unknown: -2, not_care: -1, none: 0, add_11: 1, add_s11: 2 };
  const ThirteenthTypes = { unknown: -2, not_care: -1, none: 0, add_13: 1, add_b13: 2, add_bb13: 3 };

  const BASIC_TYPES = ['.', 'maj', 'min', 'sus4', 'sus2', 'dim', 'aug', '5', '1'];

  const EXTENDED_TYPES = {
    maj6: [TriadTypes.maj, 0, 0, 0, ThirteenthTypes.add_13],
    min6: [TriadTypes.min, 0, 0, 0, ThirteenthTypes.add_13],
    '7': [TriadTypes.maj, SeventhTypes.add_b7, 0, 0, 0],
    maj7: [TriadTypes.maj, SeventhTypes.add_7, 0, 0, 0],
    min7: [TriadTypes.min, SeventhTypes.add_b7, 0, 0, 0],
    minmaj7: [TriadTypes.min, SeventhTypes.add_7, 0, 0, 0],
    dim7: [TriadTypes.dim, SeventhTypes.add_bb7, 0, 0, 0],
    hdim7: [TriadTypes.dim, SeventhTypes.add_b7, 0, 0, 0],
    '9': [TriadTypes.maj, SeventhTypes.add_b7, NinthTypes.add_9, 0, 0],
    maj9: [TriadTypes.maj, SeventhTypes.add_7, NinthTypes.add_9, 0, 0],
    min9: [TriadTypes.min, SeventhTypes.add_b7, NinthTypes.add_9, 0, 0],
    '11': [TriadTypes.maj, SeventhTypes.add_b7, NinthTypes.add_9, EleventhTypes.add_11, 0],
    min11: [TriadTypes.min, SeventhTypes.add_b7, NinthTypes.add_9, EleventhTypes.add_11, 0],
    '13': [TriadTypes.maj, SeventhTypes.add_b7, NinthTypes.add_9, EleventhTypes.add_11, ThirteenthTypes.add_13],
    maj13: [TriadTypes.maj, SeventhTypes.add_7, NinthTypes.add_9, EleventhTypes.add_11, ThirteenthTypes.add_13],
    min13: [TriadTypes.min, SeventhTypes.add_b7, NinthTypes.add_9, EleventhTypes.add_11, ThirteenthTypes.add_13],
    '': [TriadTypes.one, 0, 0, 0, 0],
    N: [TriadTypes.none, -2, -2, -2, -2],
    X: [-2, -2, -2, -2, -2],
  };

  const ADD_NOTES = {
    '7': [7, SeventhTypes.add_7],
    b7: [7, SeventhTypes.add_b7],
    bb7: [7, SeventhTypes.add_bb7],
    '2': [9, NinthTypes.add_9],
    '9': [9, NinthTypes.add_9],
    '#9': [9, NinthTypes.add_s9],
    b9: [9, NinthTypes.add_b9],
    '4': [11, EleventhTypes.add_11],
    '11': [11, EleventhTypes.add_11],
    '#11': [11, EleventhTypes.add_s11],
    '13': [13, ThirteenthTypes.add_13],
    b13: [13, ThirteenthTypes.add_b13],
    '6': [6, ThirteenthTypes.add_13],
    b6: [6, ThirteenthTypes.add_b13],
    bb6: [6, ThirteenthTypes.add_bb13],
    '#4': [5, TriadTypes.x],
    b5: [5, TriadTypes.x],
    '5': [5, TriadTypes.x],
    '#5': [5, TriadTypes.x],
    b3: [3, TriadTypes.x],
    b2: [3, TriadTypes.x],
    '3': [3, TriadTypes.x],
  };

  // dec_class -> index ke [triad, seventh, ninth, eleventh, thirteenth]
  // persis list Python: [-1,-1,-1,0,-1,0,4,1,-1,2,-1,3,-1,4]
  const DEC_INDEX_BY_CLASS = [-1, -1, -1, 0, -1, 0, 4, 1, -1, 2, -1, 3, -1, 4];

  function parseChordType(str) {
    const idx = BASIC_TYPES.indexOf(str);
    if (idx >= 0) {
      return [idx, SeventhTypes.none, NinthTypes.none, EleventhTypes.none, ThirteenthTypes.none];
    }
    if (Object.prototype.hasOwnProperty.call(EXTENDED_TYPES, str)) {
      return EXTENDED_TYPES[str].slice();
    }
    throw new Error('Unknown chord type ' + str);
  }

  function decodeSuffix(str) {
    let chordTypeStr, addNotes = [], omitNotes = [];
    if (str.indexOf('(') >= 0) {
      if (str[str.length - 1] !== ')') throw new Error('Malformed chord suffix ' + str);
      const bracketPos = str.indexOf('(');
      chordTypeStr = str.slice(0, bracketPos);
      const addOmitNotes = str.slice(bracketPos + 1, -1).split(',');
      omitNotes = addOmitNotes.filter((s) => s.charAt(0) === '*').map((s) => s.slice(1));
      addNotes = addOmitNotes.filter((s) => s.charAt(0) !== '*');
    } else {
      chordTypeStr = str;
    }
    const result = parseChordType(chordTypeStr);

    if (omitNotes.length > 0) {
      const validOmitTypes = ['1', 'b3', '3', 'b5', '5', 'b7', '7'];
      const omitFound = new Array(validOmitTypes.length).fill(false);
      for (const omitNote of omitNotes) {
        const oi = validOmitTypes.indexOf(omitNote);
        if (oi < 0) throw new Error('Invalid omit type ' + omitNote + ' in ' + str);
        omitFound[oi] = true;
      }
      if (result[0] === TriadTypes.maj && omitFound[2]) {
        result[0] = TriadTypes.power; omitFound[2] = false;
      } else if (result[0] === TriadTypes.min && omitFound[1]) {
        result[0] = TriadTypes.power; omitFound[1] = false;
      }
      if (result[0] === TriadTypes.power && omitFound[4]) {
        result[0] = TriadTypes.one; omitFound[4] = false;
      }
      if (omitFound[0] || omitFound[1] || omitFound[2] || omitFound[3] || omitFound[4]) {
        result[0] = TriadTypes.x;
      }
      if (result[1] === SeventhTypes.add_b7 && omitFound[5]) {
        result[1] = SeventhTypes.none; omitFound[5] = false;
      } else if (result[1] === SeventhTypes.add_7 && omitFound[6]) {
        result[1] = SeventhTypes.none; omitFound[6] = false;
      }
      if (omitFound[5] || omitFound[6]) {
        result[1] = SeventhTypes.unknown;
      }
    }

    for (const note of addNotes) {
      if (note === '1') {
        continue;
      } else if (note === '5' && result[0] === TriadTypes.one) {
        result[0] = TriadTypes.power;
      } else if (Object.prototype.hasOwnProperty.call(ADD_NOTES, note)) {
        const decClass = ADD_NOTES[note][0];
        const decType = ADD_NOTES[note][1];
        const decIndex = DEC_INDEX_BY_CLASS[decClass];
        if (result[decIndex] > 0 || result[decIndex] === -2) {
          result[decIndex] = -2;
        }
        result[decIndex] = decType;
      } else {
        throw new Error('Unknown decoration ' + note + ' @ ' + str);
      }
    }
    return result;
  }

  function getScaleAndSuffix(name) {
    const scaleChars = 'C*D*EF*G*A*B';
    let result = scaleChars.indexOf(name[0]);
    let prefixLength = 1;
    if (name.length > 1) {
      if (name[1] === 'b') {
        result -= 1;
        if (result < 0) result += 12;
        prefixLength = 2;
      }
      if (name[1] === '#') {
        result += 1;
        if (result >= 12) result -= 12;
        prefixLength = 2;
      }
    }
    return [result, name.slice(prefixLength)];
  }

  function scaleNameToValue(name) {
    const scaleChars = '1*2*34*5*6*78*9';
    const result = scaleChars.indexOf(name[name.length - 1]);
    const bCount = (name.match(/b/g) || []).length;
    const sharpCount = (name.match(/#/g) || []).length;
    return (((result - bCount + sharpCount + 12) % 12) + 12) % 12;
  }

  class Chord {
    constructor(name) {
      if (name.indexOf(':') >= 0) {
        const [root, rest] = getScaleAndSuffix(name);
        if (rest[0] !== ':') throw new Error('Malformed chord name ' + name);
        this.root = root;
        let suffix = rest.slice(1);
        this.bass = this.root;
        if (suffix.indexOf('/') >= 0) {
          const slashPos = suffix.indexOf('/');
          const bassStr = suffix.slice(slashPos + 1);
          this.bass = (scaleNameToValue(bassStr) + this.root) % 12;
          suffix = suffix.slice(0, slashPos);
        }
        const dec = decodeSuffix(suffix);
        [this.triad, this.seventh, this.ninth, this.eleventh, this.thirteenth] = dec;
      } else if (name === 'N') {
        this.root = -1; this.bass = -1;
        [this.triad, this.seventh, this.ninth, this.eleventh, this.thirteenth] = decodeSuffix('N');
      } else if (name === 'X') {
        this.root = -2; this.bass = -2;
        [this.triad, this.seventh, this.ninth, this.eleventh, this.thirteenth] = decodeSuffix('X');
      } else {
        throw new Error('Unknown chord name ' + name);
      }
    }

    // setara Chord.to_numpy() di Python
    toArray() {
      let triad;
      if (this.triad <= 0) triad = this.triad;
      else triad = (this.triad - 1) * 12 + 1 + this.root;
      return [triad, this.bass, this.seventh, this.ninth, this.eleventh, this.thirteenth];
    }
  }

  // setara shift_complex_chord_array() di Python
  function shiftComplexChordArray(array, shift) {
    const newArray = array.slice();
    if (newArray[0] > 0) {
      const base = Math.floor((newArray[0] - 1) / 12);
      const root = (((newArray[0] - 1 + shift) % 12) + 12) % 12;
      newArray[0] = base * 12 + root + 1;
    }
    if (newArray[1] >= 0) {
      newArray[1] = (((newArray[1] + shift) % 12) + 12) % 12;
    }
    return newArray;
  }

  const ComplexChord = {
    NUM_TO_ABS_SCALE,
    TriadTypes,
    SeventhTypes,
    NinthTypes,
    EleventhTypes,
    ThirteenthTypes,
    Chord,
    shiftComplexChordArray,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = ComplexChord;
  } else {
    global.ComplexChord = ComplexChord;
  }
})(typeof self !== 'undefined' ? self : this);
