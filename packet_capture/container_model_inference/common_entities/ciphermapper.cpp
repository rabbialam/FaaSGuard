#define _GNU_SOURCE
#include <dlfcn.h>
#include <openssl/evp.h>
#include <unistd.h>
#include <sys/socket.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <vector>

using ByteVec = std::vector<unsigned char>;

struct ByteVecHash {
    size_t operator()(const ByteVec& v) const {
        size_t h = 5381;
        for (unsigned char c : v) h = ((h << 5) + h) + c;
        return h;
    }
};

struct ByteVecEqual {
    bool operator()(const ByteVec& a, const ByteVec& b) const {
        return a == b;
    }
};

static std::unordered_map<ByteVec, ByteVec, ByteVecHash, ByteVecEqual> map;

// Hooked functions
using EncryptFunc = int(*)(EVP_CIPHER_CTX*, unsigned char*, int*, const unsigned char*, int);
using DecryptFunc = int(*)(EVP_CIPHER_CTX*, unsigned char*, int*, const unsigned char*, int);

static EncryptFunc real_encrypt = nullptr;
static DecryptFunc real_decrypt = nullptr;
static SendFunc    real_send    = nullptr;

// Hook: EVP_EncryptUpdate
extern "C"
int EVP_EncryptUpdate(EVP_CIPHER_CTX* ctx, unsigned char* out, int* outl,
                      const unsigned char* in, int inl) {
    if (!real_encrypt)
        real_encrypt = (EncryptFunc)dlsym(RTLD_NEXT, "EVP_EncryptUpdate");

    int res = real_encrypt(ctx, out, outl, in, inl);
    if (res == 1 && *outl > 0) {
        ByteVec plain(in, in + inl);
        ByteVec cipher(out, out + *outl);
        map[cipher] = plain;
    }
    return res;
}

// Hook: EVP_DecryptUpdate
extern "C"
int EVP_DecryptUpdate(EVP_CIPHER_CTX* ctx, unsigned char* out, int* outl,
                      const unsigned char* in, int inl) {
    if (!real_decrypt)
        real_decrypt = (DecryptFunc)dlsym(RTLD_NEXT, "EVP_DecryptUpdate");

    ByteVec key(in, in + inl);
    auto it = map.find(key);
    if (it != map.end()) {
        fprintf(stderr, "[DECRYPT] Plaintext: %.*s\n", (int)it->second.size(), it->second.data());
        map.erase(it);
    }

    return real_decrypt(ctx, out, outl, in, inl);
}

// Hook: send()
extern "C"
ssize_t send(int sockfd, const void* buf, size_t len, int flags) {
    if (!real_send)
        real_send = (SendFunc)dlsym(RTLD_NEXT, "send");

    ByteVec key((const unsigned char*)buf, (const unsigned char*)buf + len);
    auto it = map.find(key);
    if (it != map.end()) {
        fprintf(stderr, "[SEND] Plaintext: %.*s\n", (int)it->second.size(), it->second.data());
        map.erase(it);
    }

    return real_send(sockfd, buf, len, flags);
}

// Optional: recv() passthrough
extern "C"
ssize_t recv(int sockfd, void* buf, size_t len, int flags) {
    static auto real_recv = (ssize_t(*)(int, void*, size_t, int))dlsym(RTLD_NEXT, "recv");
    return real_recv(sockfd, buf, len, flags);
}

// Constructor
__attribute__((constructor))
static void init() {
    fprintf(stderr, "[libintercept.so] Loaded and ready.\n");
}
