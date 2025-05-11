use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::collections::HashSet;

/// BPE‐tokenize a single summary (text) given a Python list of `"first second"` merges.
#[pyfunction]
fn bpe_tokenize(text: &str, merges: &PyList) -> Vec<String> {
    // 1) Build a HashSet<(first, second)> once for O(1) membership checks.
    let merge_set: HashSet<(String, String)> = merges
        .iter()
        .filter_map(|item| item.extract::<String>().ok())
        .filter_map(|merge| {
            let mut parts = merge.split_whitespace();
            let first = parts.next()?;
            let second = parts.next()?;
            Some((first.to_string(), second.to_string()))
        })
        .collect();

    // 2) Split the text into words and collect into a Vec<String>
    let words: Vec<String> = text
        .split_whitespace()
        .map(str::to_string)
        .collect();

    // 3) Parallelize over each word
    words
        .par_iter()
        .map(|word| {
            // Start with the raw word
            let mut tokenized = word.clone();

            // Iteratively apply any merge that matches
            loop {
                let mut did_merge = false;
                for (first, second) in &merge_set {
                    if let Some(merged) = apply_merge(&tokenized, first, second) {
                        tokenized = merged;
                        did_merge = true;
                        break; // restart from the top of merge_set
                    }
                }
                if !did_merge {
                    break;
                }
            }

            tokenized
        })
        .collect()
}

/// Try to merge `first`+`second` within `token`, returning the new String if found.
fn apply_merge(token: &str, first: &str, second: &str) -> Option<String> {
    // find `first` in token
    if let Some(idx) = token.find(first) {
        let start = &token[..idx];
        let rest = &token[idx + first.len()..];
        // now find `second` in the remainder
        if let Some(idx2) = rest.find(second) {
            // build merged: start + first+second + suffix
            let merged = format!(
                "{}{}{}",
                start,
                first,
                &rest[idx2 + second.len()..]
            );
            return Some(merged);
        }
    }
    None
}

#[pymodule]
fn rust_preprocess(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bpe_tokenize, m)?)?;
    Ok(())
}
