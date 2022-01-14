// no need to #include anything, this file is used internally by to_object_arrays.pyx

typedef struct {
    PyObject_VAR_HEAD

    // asyncpg specifics begin here
    // if they add another field, we will break spectacularly
    Py_hash_t self_hash;
    void *desc;  // we don't care of the type
    PyObject *ob_item[1];  // embedded in the tail, the count matches len()
} ApgRecordObject;

#define ApgRecord_GET_ITEM(op, i) (((ApgRecordObject *)(op))->ob_item[i])
