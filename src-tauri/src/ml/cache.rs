//! Caching for ML inference results.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cache entry with timestamp.
struct CacheEntry<T> {
    value: T,
    created_at: Instant,
}

/// LRU-style cache for inference results.
pub struct InferenceCache<T> {
    entries: RwLock<HashMap<String, CacheEntry<T>>>,
    max_age: Duration,
    max_entries: usize,
}

impl<T: Clone> InferenceCache<T> {
    /// Create a new cache with specified max age and size.
    pub fn new(max_age: Duration, max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_age,
            max_entries,
        }
    }

    /// Get a cached value if it exists and hasn't expired.
    pub fn get(&self, key: &str) -> Option<T> {
        let entries = self.entries.read();

        entries.get(key).and_then(|entry| {
            if entry.created_at.elapsed() < self.max_age {
                Some(entry.value.clone())
            } else {
                None
            }
        })
    }

    /// Store a value in the cache.
    pub fn set(&self, key: String, value: T) {
        let mut entries = self.entries.write();

        // Remove expired entries if at capacity
        if entries.len() >= self.max_entries {
            self.cleanup_expired(&mut entries);
        }

        // If still at capacity, remove oldest entry
        if entries.len() >= self.max_entries {
            if let Some(oldest_key) = self.find_oldest(&entries) {
                entries.remove(&oldest_key);
            }
        }

        entries.insert(
            key,
            CacheEntry {
                value,
                created_at: Instant::now(),
            },
        );
    }

    /// Remove a specific entry.
    pub fn remove(&self, key: &str) {
        self.entries.write().remove(key);
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Get current cache size.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Remove expired entries.
    fn cleanup_expired(&self, entries: &mut HashMap<String, CacheEntry<T>>) {
        let now = Instant::now();
        entries.retain(|_, entry| now.duration_since(entry.created_at) < self.max_age);
    }

    /// Find the oldest entry key.
    fn find_oldest(&self, entries: &HashMap<String, CacheEntry<T>>) -> Option<String> {
        entries
            .iter()
            .min_by_key(|(_, entry)| entry.created_at)
            .map(|(key, _)| key.clone())
    }
}

impl<T: Clone> Default for InferenceCache<T> {
    fn default() -> Self {
        Self::new(Duration::from_secs(300), 100) // 5 minutes, 100 entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_cache_basic() {
        let cache: InferenceCache<Vec<f64>> = InferenceCache::default();

        cache.set("test".to_string(), vec![1.0, 2.0, 3.0]);

        let result = cache.get("test");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cache_miss() {
        let cache: InferenceCache<Vec<f64>> = InferenceCache::default();
        let result = cache.get("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_expiry() {
        let cache: InferenceCache<String> = InferenceCache::new(
            Duration::from_millis(50),
            10,
        );

        cache.set("test".to_string(), "value".to_string());

        // Should exist immediately
        assert!(cache.get("test").is_some());

        // Wait for expiry
        sleep(Duration::from_millis(60));

        // Should be expired
        assert!(cache.get("test").is_none());
    }

    #[test]
    fn test_cache_max_entries() {
        let cache: InferenceCache<i32> = InferenceCache::new(
            Duration::from_secs(300),
            3,
        );

        cache.set("a".to_string(), 1);
        cache.set("b".to_string(), 2);
        cache.set("c".to_string(), 3);

        assert_eq!(cache.len(), 3);

        // Adding fourth should evict oldest
        cache.set("d".to_string(), 4);

        assert_eq!(cache.len(), 3);
    }
}
