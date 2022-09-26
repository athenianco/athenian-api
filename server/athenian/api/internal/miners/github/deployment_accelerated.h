template <typename T>
inline void hash_combine(size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct PullRequestAddr {
  uint32_t di;
  uint32_t ri;
  uint64_t pri;

  bool operator==(const PullRequestAddr &other) const {
    return di == other.di && ri == other.ri && pri == other.pri;
  }
};

struct ReleaseAddr {
  uint32_t di;
  uint32_t ri;
  
  bool operator==(const ReleaseAddr &other) const {
    return di == other.di && ri == other.ri;
  }
};

namespace std {
  template <>
  struct hash<PullRequestAddr> {
    size_t operator()(const PullRequestAddr &k) const {
      size_t seed = 0;
      hash_combine(seed, k.di);
      hash_combine(seed, k.ri);
      hash_combine(seed, k.pri);
      return seed;
    }
  };
  
  template <>
  struct hash<ReleaseAddr> {
    size_t operator()(const ReleaseAddr &k) const {
      return std::hash<uint64_t>()(*reinterpret_cast<const uint64_t *>(&k));
    }
  };
}
