#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <dlfcn.h>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <errno.h>
#include <vector>
#include <memory>
#include "packetanalyzer.h"
#include "evaluate.h"
#include <sys/stat.h>
#include <stdio.h>
#include <thread>
#include <pybind11/embed.h>
#include <chrono>
#define _GNU_SOURCE
#include <dlfcn.h>
#include <openssl/evp.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>

#define MAX_ENTRIES 1024
#define MAX_BUF_LEN 4096

struct Mapping {
    int socket_fd;
    unsigned char cipher[MAX_BUF_LEN];
    int clen;
    unsigned char plain[MAX_BUF_LEN];
    int plen;
    bool valid;
};

static Mapping map[MAX_ENTRIES];

// Store mapping
void store_mapping(int fd, const unsigned char* cipher, int clen, const unsigned char* plain, int plen) {
    for (int i = 0; i < MAX_ENTRIES; ++i) {
        if (!map[i].valid) {
            map[i].socket_fd = fd;
            memcpy(map[i].cipher, cipher, clen);
            map[i].clen = clen;
            memcpy(map[i].plain, plain, plen);
            map[i].plen = plen;
            map[i].valid = true;
            break;
        }
    }
}

// Retrieve and remove mapping by ciphertext
bool lookup_plain_by_cipher(const unsigned char* cipher, int clen, unsigned char* out_plain, int& out_plen) {
    for (int i = 0; i < MAX_ENTRIES; ++i) {
        if (map[i].valid && map[i].clen == clen && memcmp(map[i].cipher, cipher, clen) == 0) {
            memcpy(out_plain, map[i].plain, map[i].plen);
            out_plen = map[i].plen;
            map[i].valid = false;
            return true;
        }
    }
    return false;
}

// Hooked functions
using EncryptFunc = int(*)(EVP_CIPHER_CTX*, unsigned char*, int*, const unsigned char*, int);
using DecryptFunc = int(*)(EVP_CIPHER_CTX*, unsigned char*, int*, const unsigned char*, int);

static EncryptFunc real_encrypt = nullptr;
static DecryptFunc real_decrypt = nullptr;
static SendFunc    real_send    = nullptr;

namespace py = pybind11;

// Static variable to hold the log file
static std::ofstream log_fd;
static ssize_t (*real_write)(int, const void *, size_t) = nullptr;
using namespace std;
std::string aggrigate = "";
int incomming_socket = -1;
// Helper function to get thread ID
pid_t gettid() {
    return syscall(SYS_gettid);
}

// Initialization of the shared library (constructor)
void __attribute__((constructor)) initLibrary(void) {
    pid_t pid = getpid();
    if(pid ==1) return;
    

    static py::scoped_interpreter guard{}; 
    initilize_evaluation();
    std::string init_message = "Shared library is loaded\n";
    cout << init_message <<"PID is "<<pid<<"\n";
   // log_fd.flush();
}

// Cleanup of the shared library (destructor)
void __attribute__((destructor)) cleanupLibrary(void) {
    if (log_fd.is_open()) {
        log_fd.close();
    }
}


int is_socket(int fd) {
    struct stat statbuf;
    if (fstat(fd, &statbuf) == -1) {
        perror("fstat failed");
        return -1;
    }
    return S_ISSOCK(statbuf.st_mode);
}
// Logging function
void logMessage(int fd, const std::string &action, const void *buf, size_t len) {
    int type;
    socklen_t type_len = sizeof(type);
    if (!is_socket(fd)) {
        return;  // Not a socket, skip logging
    }

    if (type == AF_UNIX) {
        std::cout << "Unix socket\n";
        //return;
    } else if (type == AF_INET || type == AF_INET6) {
        std::cout << "TCP socket\n";
    }

    //std::cout << "In " << action << " function with fd "<<fd<<"\n ";

    // Convert buffer to hexadecimal string
    const unsigned char *p = static_cast<const unsigned char *>(buf);
    std::string http_str(reinterpret_cast<const char*>(p),len);

    auto start = std::chrono::high_resolution_clock::now();

    std::string packet =  parse_http_packet(http_str,fd,0,Type::OUT);
    //std::cout <<  packet<<" packet\n";
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    //std::cout << "HTTP packet to sentence time: " << duration.count() << " milli seconds" << std::endl;


    aggrigate +=packet;
    //float score = tokenize_sentence(aggrigate); 
    //std::cout <<"Score: "<<score <<endl;
    if(!action.compare("write") && incomming_socket==fd){
       // std::cout <<aggrigate<<"\n";
       auto start_s = std::chrono::high_resolution_clock::now();
        float score = tokenize_sentence(aggrigate);
        auto end_s = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration_S = end_s - start_s;

        std::cout << "POST score generation time  " << duration_S.count() << "ms and score is "<<score << std::endl;
        
        //cout<<"Calculated scire is: "<<score<<endl;
        aggrigate ="";
        incomming_socket = -1;
    }
    if(!action.compare("read") && incomming_socket==-1) {
        incomming_socket = fd;
    }

}


ssize_t send(int sockfd, const void *buf, size_t len, int flags) {
   // printf("in send function\n");
    static ssize_t (*real_send)(int, const void *, size_t, int) = NULL;
    if (!real_send) {
        real_send = (ssize_t (*)(int, const void *, size_t, int))dlsym(RTLD_NEXT, "send");
    }
    logMessage(sockfd, "write", buf, len);
    ssize_t result = real_send(sockfd, buf, len, flags);
    
    return result;
}

//  Name mangling creates unique names in the compiled code, which helps the linker differentiate between overloaded functions.
// For that we have to use extern C
extern "C" ssize_t writev(int fd, const struct iovec *iov, int iovcnt) {
   // printf("in writev function\n");
    static ssize_t (*real_writev)(int, const struct iovec *, int) = NULL;
    if (!real_writev) {
        real_writev = (ssize_t (*)(int, const struct iovec *, int))dlsym(RTLD_NEXT, "writev");
    }
    
    // Calculate the total length of the data
    size_t total_len = 0;
    for (int i = 0; i < iovcnt; i++) {
        total_len += iov[i].iov_len;
    }
    
    // Allocate a buffer to copy the data
    char *buffer = (char *)malloc(total_len);
    if (!buffer) {
        perror("malloc");
       // exit(EXIT_FAILURE);
    }
    
    // Copy the data into the buffer
    char *ptr = buffer;
    for (int i = 0; i < iovcnt; i++) {
        memcpy(ptr, iov[i].iov_base, iov[i].iov_len);
        ptr += iov[i].iov_len;
    }
    
    // Logging function (implement this as needed)
   
    unsigned char plain[MAX_BUF_LEN];
    int plen = 0;
    if (lookup_plain_by_cipher((const unsigned char*)buffer, (int)len, plain, plen)) {
        logMessage(fd, "write", plain, plen);
       
    }
    
   
    
    // Clean up
    free(buffer);
    
    // Call the original writev function
    ssize_t result = real_writev(fd, iov, iovcnt);
    
    return result;
}

ssize_t sendmsg(int sockfd,const struct msghdr *msg, int flags) {
      //printf("in sendmsg function\n");
    static ssize_t (*real_sendmsg)(int, const struct msghdr *,  int) = NULL;
    if (!real_sendmsg) {
        real_sendmsg = (ssize_t (*)(int, const struct msghdr *,  int))dlsym(RTLD_NEXT, "sendmsg");
    }
     size_t len = 0;
    for (size_t i = 0; i < msg->msg_iovlen; i++) {
        len += msg->msg_iov[i].iov_len;
    }
    
    // Allocate a buffer to copy the data
    char *buf = (char *)malloc(len);
    if (!buf) {
        perror("malloc");
        //exit(EXIT_FAILURE);
    }
    
    // Copy the data into the buffer
    char *ptr = buf;
    for (size_t i = 0; i < msg->msg_iovlen; i++) {
        memcpy(ptr, msg->msg_iov[i].iov_base, msg->msg_iov[i].iov_len);
        ptr += msg->msg_iov[i].iov_len;
    }
    unsigned char plain[MAX_BUF_LEN];
    int plen = 0;
    if (lookup_plain_by_cipher((const unsigned char*)buf, (int)len, plain, plen)) {
        logMessage(fd, "write", plain, plen);
       
    }    ssize_t result = real_sendmsg(sockfd, msg, flags);
    
    return result;
}

ssize_t recv(int sockfd, void *buf, size_t len, int flags) {
    static ssize_t (*real_recv)(int, void *, size_t, int) = NULL;
    if (!real_recv) {
        real_recv = (ssize_t (*)(int, void *, size_t, int))dlsym(RTLD_NEXT, "recv");
    }
    ssize_t result = real_recv(sockfd, buf, len, flags);
  
    if (result > 0) {
        store_mapping(sockfd, (const unsigned char*)buf, (int)n, nullptr, 0);
    }
    //logMessage(sockfd, "read", buf, result);
    return result;
}

ssize_t read(int fd, void *buf, size_t count) {
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = (ssize_t (*)(int, void *, size_t))dlsym(RTLD_NEXT, "read");
    }
    ssize_t result = real_read(fd, buf, count);
    if (result > 0) {
        store_mapping(sockfd, (const unsigned char*)buf, (int)n, nullptr, 0);
    }
    //logMessage(fd, "read", buf, result);
    return result;
}

ssize_t write(int fd, const void *buf, size_t count) {
    if (!real_write) {
        real_write = (ssize_t (*)(int, const void *, size_t))dlsym(RTLD_NEXT, "write");
    }
    //printf("in write message with fd %d\n",fd);
    unsigned char plain[MAX_BUF_LEN];
    int plen = 0;
    if (lookup_plain_by_cipher((const unsigned char*)buf, (int)count, plain, plen)) {
        logMessage(fd, "write", plain, plen);
       
    }
    ssize_t result = real_write(fd, buf, count);

    
    return result;
}
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen) {
      //printf("in sendto function\n");            
    static ssize_t (*real_sendto)(int, const void *, size_t, int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_sendto) {
        real_sendto = (ssize_t (*)(int, const void *, size_t, int, const struct sockaddr *, socklen_t))dlsym(RTLD_NEXT, "sendto");
    }
   
    unsigned char plain[MAX_BUF_LEN];
    int plen = 0;
    if (lookup_plain_by_cipher((const unsigned char*)buf, (int)len, plain, plen)) {
        logMessage(sockfd, "write", plain, plen);
       
    }
    return real_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen) {
    static ssize_t (*real_recvfrom)(int, void *, size_t, int, struct sockaddr *, socklen_t *) = NULL;
    if (!real_recvfrom) {
        real_recvfrom = (ssize_t (*)(int, void *, size_t, int, struct sockaddr *, socklen_t *))dlsym(RTLD_NEXT, "recvfrom");
    }
    ssize_t result = real_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    if (result > 0) {
        store_mapping(sockfd, (const unsigned char*)buf, (int)n, nullptr, 0);
    }
    //logMessage(sockfd, "read", buf, result);
    return result;
}

// Hook: EVP_EncryptUpdate
extern "C"
int EVP_EncryptUpdate(EVP_CIPHER_CTX* ctx, unsigned char* out, int* outl,
                      const unsigned char* in, int inl) {
    if (!real_encrypt)
        real_encrypt = (EncryptFunc)dlsym(RTLD_NEXT, "EVP_EncryptUpdate");

    int res = real_encrypt(ctx, out, outl, in, inl);
    if (res == 1 && *outl > 0) {
        store_mapping(-1, out, *outl, in, inl);  // -1 for unknown socket
    }
    return res;
}

extern "C"
int EVP_DecryptUpdate(EVP_CIPHER_CTX* ctx, unsigned char* out, int* outl,
                      const unsigned char* in, int inl) {
    if (!real_decrypt)
        real_decrypt = (DecryptFunc)dlsym(RTLD_NEXT, "EVP_DecryptUpdate");

    int res = real_decrypt(ctx, out, outl, in, inl);
    if (res == 1 && *outl > 0) {
        // Update matching recv entry with plaintext
        for (int i = 0; i < MAX_ENTRIES; ++i) {
            if (map[i].valid && map[i].clen == inl && memcmp(map[i].cipher, in, inl) == 0) {
                memcpy(map[i].plain, out, *outl);
                map[i].plen = *outl;
                 logMessage(map[i].socket_fd, "read", out, outl);
               // fprintf(stderr, "[DECRYPT] fd=%d, Plaintext: %.*s\n", map[i].socket_fd, *outl, out);
                break;
            }
        }
    }
    return res;
}
