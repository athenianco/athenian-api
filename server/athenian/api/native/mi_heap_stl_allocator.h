#include <memory>
#include <new>

#include <mimalloc.h>

#define mi_unlikely(x)     __builtin_expect(!!(x),false)
#define mi_likely(x)       __builtin_expect(!!(x),true)
#define mi_decl_inline     inline __attribute__((always_inline))
#define mi_decl_noinline   __attribute__((noinline))

namespace {
  mi_decl_inline bool mi_mul_overflow(size_t count, size_t size, size_t* total) {
    #if (SIZE_MAX == ULONG_MAX)
      return __builtin_umull_overflow(count, size, (unsigned long *)total);
    #elif (SIZE_MAX == UINT_MAX)
      return __builtin_umul_overflow(count, size, (unsigned int *)total);
    #else
      return __builtin_umulll_overflow(count, size, (unsigned long long *)total);
    #endif
  }
  
  mi_decl_inline bool mi_count_size_overflow(size_t count, size_t size, size_t* total) {
    if (count==1) {  // quick check for the case where count is one (common for C++ allocators)
      *total = size;
      return false;
    }
    else if (mi_unlikely(mi_mul_overflow(count, size, total))) {
      *total = SIZE_MAX;
      return true;
    }
    else return false;
  }

  mi_decl_inline bool mi_try_new_handler() {
   std::new_handler h = std::get_new_handler();
    if (h==NULL) {
      throw std::bad_alloc();
      return false;
    } else {
      h();
      return true;
    }
  }
  
  mi_decl_restrict void* mi_heap_new_n(size_t count, size_t size, mi_heap_t *heap) {
    size_t total;
    if (mi_unlikely(mi_count_size_overflow(count, size, &total))) {
      mi_try_new_handler();  // on overflow we invoke the try_new_handler once to potentially throw std::bad_alloc
      return NULL;
    }
    else {
      void* p = NULL;
      do {
        p = mi_heap_malloc(heap, total);
      } while (mi_unlikely(p == NULL) && mi_try_new_handler());
      return p;
    }
  }
}

template<class T> struct mi_heap_stl_allocator {
  typedef T                 value_type;
  typedef std::size_t       size_type;
  typedef std::ptrdiff_t    difference_type;
  typedef value_type&       reference;
  typedef value_type const& const_reference;
  typedef value_type*       pointer;
  typedef value_type const* const_pointer;
  template <class U> struct rebind { typedef mi_heap_stl_allocator<U> other; };

  mi_heap_stl_allocator() {
    mi_heap_t *heap = mi_heap_new();
    this->_heap.reset(new(static_cast<managed_heap *>(mi_heap_new_n(1, sizeof(managed_heap), heap))) managed_heap(heap), managed_heap::destroy);
  }
  mi_heap_stl_allocator(const mi_heap_stl_allocator&) mi_attr_noexcept = default;
  template<class U> mi_heap_stl_allocator(const mi_heap_stl_allocator<U>& other) mi_attr_noexcept : _heap(std::reinterpret_pointer_cast<mi_heap_stl_allocator<T>::managed_heap>(other._heap)) { }
  mi_heap_stl_allocator  select_on_container_copy_construction() const { return *this; }
  void              deallocate(T* p, size_type) { if (_heap->free_enabled) mi_free(p); }

  #if (__cplusplus >= 201703L)  // C++17
  mi_decl_nodiscard T* allocate(size_type count) { return static_cast<T*>(mi_heap_new_n(count, sizeof(T), _heap->heap)); }
  mi_decl_nodiscard T* allocate(size_type count, const void*) { return allocate(count); }
  #else
  mi_decl_nodiscard pointer allocate(size_type count, const void* = 0) { return static_cast<pointer>(mi_heap_new_n(count, sizeof(value_type), _heap->heap)); }
  #endif

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap            = std::true_type;
  using is_always_equal                        = std::true_type;
  template <class U, class ...Args> void construct(U* p, Args&& ...args) { ::new(p) U(std::forward<Args>(args)...); }
  template <class U> void destroy(U* p) mi_attr_noexcept { p->~U(); }

  size_type     max_size() const mi_attr_noexcept { return (PTRDIFF_MAX/sizeof(value_type)); }
  pointer       address(reference x) const        { return &x; }
  const_pointer address(const_reference x) const  { return &x; }

  void enable_free() mi_attr_noexcept { this->_heap->free_enabled = true; }
  void disable_free() mi_attr_noexcept { this->_heap->free_enabled = false; }
  void collect(bool force = false) mi_attr_noexcept { mi_heap_collect(_heap->heap, force); }

  protected:
    struct managed_heap {
      managed_heap(mi_heap_t *heap): heap(heap), free_enabled(true) { }
      managed_heap(const managed_heap&) = delete;
      managed_heap& operator=(managed_heap const&) = delete;
      ~managed_heap() = delete;
      static void destroy(managed_heap *ptr) { mi_heap_destroy(ptr->heap); }

      mi_heap_t *heap;
      bool free_enabled;
    };

    std::shared_ptr<managed_heap> _heap;

    template <typename>
    friend struct mi_heap_stl_allocator;
    template<class T1,class T2>
    friend bool operator==(const mi_heap_stl_allocator<T1>& first, const mi_heap_stl_allocator<T2>& second) mi_attr_noexcept;
    template<class T1,class T2>
    friend bool operator!=(const mi_heap_stl_allocator<T1>& first, const mi_heap_stl_allocator<T2>& second) mi_attr_noexcept;
};

template<class T1,class T2> bool operator==(const mi_heap_stl_allocator<T1>& first, const mi_heap_stl_allocator<T2>& second) mi_attr_noexcept { return first._heap == second._heap; }
template<class T1,class T2> bool operator!=(const mi_heap_stl_allocator<T1>& first, const mi_heap_stl_allocator<T2>& second) mi_attr_noexcept { return first._heap != second._heap; }

#include <unordered_map>
#include <unordered_set>
#include <vector>

template<
    class T,
    class U,
    class HASH = std::hash<T>,
    class PRED = std::equal_to<T>
>
using mi_unordered_map = std::unordered_map<T, U, HASH, PRED, mi_heap_stl_allocator<std::pair<const T, U>>>;

template<
    class T,
    class HASH = std::hash<T>,
    class PRED = std::equal_to<T>
>
using mi_unordered_set = std::unordered_set<T, HASH, PRED, mi_heap_stl_allocator<T>>;

template<class T>
using mi_vector = std::vector<T, mi_heap_stl_allocator<T>>;