/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 The Viv authors */
/* fstrace: the files a process looks at, for the gate's cache (tools/gate/cache.py).
 *
 * Loaded with DYLD_INSERT_LIBRARIES (macOS), it appends to $VIV_TRACE_INPUTS,
 * in lang/viv/src/trace.rs's format, every path the process opens, stats,
 * or lists:
 *
 *     f <TAB> <absolute path> <TAB> 1|0     opened or looked at; 1 if it was there
 *     w <TAB> <absolute path> <TAB> 1       opened for writing: an output (it is an f line too)
 *     l <TAB> <absolute directory>          its entries were listed (their names are an input)
 *     x <TAB> <program> <TAB> <args...>     a program it started
 *     u <TAB> <program>                     one it started without this library (a sanitizer build)
 *
 * Children inherit the library, except programs macOS protects (/bin/sh,
 * /usr/bin/...): dyld drops DYLD_* for them and for everything they start.
 * cache.py therefore refuses to cache a run that started a shell or
 * another protected program it does not know reads only what its
 * arguments name (cc, clang, leaks).
 *
 * Built by cache.py: cc -dynamiclib -O2 -o target/gate/fstrace.dylib fstrace.c
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

#define INTERPOSE(rep, orig) \
    __attribute__((used)) static struct { const void *r; const void *o; } _i_##orig \
    __attribute__((section("__DATA,__interpose"))) = { (const void *)(unsigned long)&rep, (const void *)(unsigned long)&orig };

static void emit(const char *line, size_t n) {
    const char *sink = getenv("VIV_TRACE_INPUTS");
    if (!sink || !*sink) return;
    int fd = open(sink, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return;
    (void)write(fd, line, n);
    close(fd);
}

/* the absolute form of `path`, relative to the directory `dirfd` names */
static void absolute(int dirfd, const char *path, char *out) {
    if (path[0] == '/') { strlcpy(out, path, MAXPATHLEN); return; }
    char base[MAXPATHLEN] = "";
    if (dirfd == AT_FDCWD) { if (!getcwd(base, sizeof base)) base[0] = 0; }
    else if (fcntl(dirfd, F_GETPATH, base) == -1) base[0] = 0;
    snprintf(out, MAXPATHLEN, "%s/%s", base, path);
}

static void note_kind(int dirfd, const char *path, int ok, const char *kind) {
    if (!path) return;
    int e = errno;
    char abs[MAXPATHLEN], line[MAXPATHLEN + 16];
    absolute(dirfd, path, abs);
    int n = snprintf(line, sizeof line, "%s\t%s\t%d\n", kind, abs, ok ? 1 : 0);
    if (n > 0 && n < (int)sizeof line) emit(line, (size_t)n);
    errno = e;
}

static void note(int dirfd, const char *path, int ok) { note_kind(dirfd, path, ok, "f"); }

/* an open for writing is also an output: `w` */
static void note_open(int dirfd, const char *path, int flags, int ok) {
    note(dirfd, path, ok);
    if (ok && (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND))) note_kind(dirfd, path, 1, "w");
}

static int my_open(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = (mode_t)va_arg(ap, int); va_end(ap); }
    int r = open(path, flags, mode);
    note_open(AT_FDCWD, path, flags, r >= 0);
    return r;
}
INTERPOSE(my_open, open)

static int my_openat(int dirfd, const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = (mode_t)va_arg(ap, int); va_end(ap); }
    int r = openat(dirfd, path, flags, mode);
    note_open(dirfd, path, flags, r >= 0);
    return r;
}
INTERPOSE(my_openat, openat)

static int my_stat(const char *path, struct stat *st) { int r = stat(path, st); note(AT_FDCWD, path, r == 0); return r; }
INTERPOSE(my_stat, stat)
static int my_lstat(const char *path, struct stat *st) { int r = lstat(path, st); note(AT_FDCWD, path, r == 0); return r; }
INTERPOSE(my_lstat, lstat)
static int my_fstatat(int dirfd, const char *path, struct stat *st, int flag) {
    int r = fstatat(dirfd, path, st, flag);
    if (path && *path) note(dirfd, path, r == 0);
    return r;
}
INTERPOSE(my_fstatat, fstatat)
static int my_access(const char *path, int mode) { int r = access(path, mode); note(AT_FDCWD, path, r == 0); return r; }
INTERPOSE(my_access, access)

static DIR *my_opendir(const char *path) {
    DIR *d = opendir(path);
    if (d) {
        int e = errno;
        char abs[MAXPATHLEN], line[MAXPATHLEN + 8];
        absolute(AT_FDCWD, path, abs);
        int n = snprintf(line, sizeof line, "l\t%s\n", abs);
        if (n > 0 && n < (int)sizeof line) emit(line, (size_t)n);
        errno = e;
    } else {
        note(AT_FDCWD, path, 0);
    }
    return d;
}
INTERPOSE(my_opendir, opendir)

/* A sanitizer runtime (ASan, UBSan, TSan) interposes the same functions, and the two
 * spin against each other: such a program is started without this library, and the
 * run is marked untraced (`u`), which cache.py does not cache. */
static int sanitized(const char *file) {
    char path[MAXPATHLEN];
    if (!file) return 0;
    if (strchr(file, '/')) strlcpy(path, file, sizeof path);
    else {
        const char *env = getenv("PATH");
        char dirs[8192];
        strlcpy(dirs, env ? env : "/usr/bin:/bin", sizeof dirs);
        int found = 0;
        char *save = NULL;
        for (char *d = strtok_r(dirs, ":", &save); d; d = strtok_r(NULL, ":", &save)) {
            snprintf(path, sizeof path, "%s/%s", d, file);
            if (access(path, X_OK) == 0) { found = 1; break; }
        }
        if (!found) return 0;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    static char buf[65536];
    ssize_t n = read(fd, buf, sizeof buf);
    close(fd);
    for (ssize_t i = 0; i + 13 < n; i++)
        if (buf[i] == 'l' && memcmp(buf + i, "libclang_rt.", 12) == 0) return 1;
    return 0;
}

/* envp without DYLD_INSERT_LIBRARIES (a static copy: the caller execs or spawns at once) */
static char *const *without_us(char *const envp[]) {
    static char *out[4096];
    int k = 0;
    for (int i = 0; envp && envp[i] && k < 4095; i++)
        if (strncmp(envp[i], "DYLD_INSERT_LIBRARIES=", 22) != 0) out[k++] = envp[i];
    out[k] = NULL;
    return out;
}

static void note_untraced(const char *path) {
    char line[MAXPATHLEN + 8];
    int n = snprintf(line, sizeof line, "u\t%s\n", path);
    if (n > 0 && n < (int)sizeof line) emit(line, (size_t)n);
}

static void note_exec(const char *path, char *const argv[]) {
    int e = errno;
    char line[8192];
    size_t n = (size_t)snprintf(line, sizeof line, "x\t%s", path ? path : "?");
    for (int i = 1; argv && argv[i] && n < sizeof line - 2; i++) {
        int k = snprintf(line + n, sizeof line - n, "\t%s", argv[i]);
        if (k < 0) break;
        n += (size_t)k;
    }
    if (n > sizeof line - 2) n = sizeof line - 2;
    line[n++] = '\n';
    emit(line, n);
    errno = e;
}

extern char **environ;

static int my_execve(const char *path, char *const argv[], char *const envp[]) {
    note_exec(path, argv);
    if (sanitized(path)) { note_untraced(path); return execve(path, argv, without_us(envp)); }
    return execve(path, argv, envp);
}
INTERPOSE(my_execve, execve)

static int my_execv(const char *path, char *const argv[]) {
    note_exec(path, argv);
    if (sanitized(path)) { note_untraced(path); unsetenv("DYLD_INSERT_LIBRARIES"); }
    return execv(path, argv);
}
INTERPOSE(my_execv, execv)
static int my_execvp(const char *file, char *const argv[]) {
    note_exec(file, argv);
    if (sanitized(file)) { note_untraced(file); unsetenv("DYLD_INSERT_LIBRARIES"); }
    return execvp(file, argv);
}
INTERPOSE(my_execvp, execvp)

static int my_posix_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *fa,
                          const posix_spawnattr_t *at, char *const argv[], char *const envp[]) {
    note_exec(path, argv);
    if (sanitized(path)) { note_untraced(path); return posix_spawn(pid, path, fa, at, argv, without_us(envp)); }
    return posix_spawn(pid, path, fa, at, argv, envp);
}
INTERPOSE(my_posix_spawn, posix_spawn)

static int my_posix_spawnp(pid_t *pid, const char *file, const posix_spawn_file_actions_t *fa,
                           const posix_spawnattr_t *at, char *const argv[], char *const envp[]) {
    note_exec(file, argv);
    if (sanitized(file)) { note_untraced(file); return posix_spawnp(pid, file, fa, at, argv, without_us(envp)); }
    return posix_spawnp(pid, file, fa, at, argv, envp);
}
INTERPOSE(my_posix_spawnp, posix_spawnp)
