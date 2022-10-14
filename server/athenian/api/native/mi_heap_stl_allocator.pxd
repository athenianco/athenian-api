from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport pair, unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector


cdef extern from "mi_heap_stl_allocator.h" nogil:
    cdef cppclass mi_heap_stl_allocator[T]:
        mi_heap_stl_allocator() except +
        mi_heap_stl_allocator(const mi_heap_stl_allocator &)
        void disable_free()
        void enable_free()

    cdef cppclass mi_unordered_map[T, U, HASH=*, PRED=*](unordered_map[T, U, HASH, PRED]):
        mi_unordered_map mi_unordered_map[X](mi_heap_stl_allocator[X]&) except +
        pair[mi_unordered_map.iterator, bool] try_emplace(...) except +
        mi_heap_stl_allocator[T] get_allocator()

    cdef cppclass mi_unordered_set[T, HASH=*, PRED=*](unordered_set[T, HASH, PRED]):
        mi_unordered_set mi_unordered_set[X](mi_heap_stl_allocator[X]&) except +
        pair[mi_unordered_set.iterator, bool] emplace(...) except +
        mi_heap_stl_allocator[T] get_allocator()

        mi_unordered_set.iterator erase(mi_unordered_set.iterator)
        mi_unordered_set.iterator erase(mi_unordered_set.iterator, mi_unordered_set.iterator)
        size_t erase(T&)

    cdef cppclass mi_vector[T](vector[T]):
        mi_vector mi_vector[X](mi_heap_stl_allocator[X]&) except +
        T& emplace_back(...) except +
        mi_heap_stl_allocator[T] get_allocator()

    cdef cppclass mi_string(string):
        mi_string mi_string[X](const char *, size_t, mi_heap_stl_allocator[X]&) except +
        mi_heap_stl_allocator[char] get_allocator()
