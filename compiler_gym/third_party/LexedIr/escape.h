// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//=============================================================================
// Escape and unescape strings.
#pragma once

#include <string>

/// Get the hex value of a character
static int char_to_hex(int c) {
  if (c >= '0' and c <= '9')
    return c - '0';
  else
    return c - 'A' + 10;
}

/// Escape a string
std::string escape(const std::string& s, bool escape_quote = false) {
  std::string t;
  t.reserve(s.size());
  char const* const hexdig = "0123456789ABCDEF";
  for (unsigned char c : s) {
    if (isprint(c) and c != '\\' and (!escape_quote or c != '"')) {
      t.push_back(c);
    } else {
      t.push_back('\\');
      switch (c) {
        default: {
          t.push_back(hexdig[c >> 4]);
          t.push_back(hexdig[c & 0xF]);
        }
      }
    }
  }
  return t;
}

/// Unescape a string
std::string unescape(const std::string& s) {
  const char* p = s.c_str();
  const char* q = p + s.size();

  std::string t;
  t.reserve(s.size());
  while (p < q) {
    if (*p != '\\') {
      t.push_back(*p);
    } else {
      // Unescape
      unsigned char c = 0;
      c |= (char_to_hex(*++p) << 4);
      c |= char_to_hex(*++p);
      t.push_back(c);
    }
    p++;
  }
  return t;
}
