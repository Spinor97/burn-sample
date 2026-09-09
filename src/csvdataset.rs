use std::sync::Arc;

use memmap2::Mmap;

struct CsvItem {
    pub features: Vec<f32>,
    pub label: f32,
}

pub struct CsvDataSet {
    mmap: Arc<Mmap>,
    line_offset: Vec<usize>,
    num_features: usize,
}

impl CsvDataSet {
    pub fn new(path: &str, num_features: usize) -> Self {
        let file = std::fs::File::open(path).expect("Failed to open");
        let mmap = Arc::new( unsafe {
            Mmap::map(&file).expect("Failed mmap")
        });

        let bytes = &mmap[..];
        let mut offsets = Vec::new();

        let mut i = 0;
        let l = bytes.len();

        while i < l && bytes[i] != b'\n' {
            i += 1;
        }

        i += 1;

        while i < l {
            if bytes[i] != b'\n' && bytes[i] != b'\r' {
                offsets.push(i);
            }
            while i < l && bytes[i] != b'\n' {
                i += 1;
            }
            
            i += 1;
        }

        if let Some(&lst) = offsets.last() {
            if lst >= l || mmap[lst..].iter().all(|&b| b == b'\n' || b == b'\r') {
                offsets.pop();
            }
        }

        Self {
            mmap,
            line_offset: offsets,
            num_features
        }
    }
}