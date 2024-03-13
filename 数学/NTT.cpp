#define __SSE__ 1
#define __SSE2__ 1
#define __SSE2_MATH__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSSE3__ 1
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,popcnt,tune=native")
#include <immintrin.h>
#include <emmintrin.h>
#include <bits/stdc++.h>
using namespace std;
 
const int mod = 998244353;
const int G = 3;
namespace NTT {
	using i32 = int32_t;
	using u32 = uint32_t;
	using i64 = int64_t;
	using u64 = uint64_t;
	using word = __m256i;
 
	inline i32 add(i32 a, i32 b) {
		return (a += b) >= mod ? a - mod : a;
	}
	inline i32 sub(i32 a, i32 b) {
		return (a -= b) < 0 ? a + mod : a;
	}
	inline i32 power(i32 a, i32 b) {
		i32 res = 1;
		for(; b; b >>= 1, a = (i64) a * a % mod)
			if(b & 1) res = (i64) res * a % mod;
		return res;
	}
 
	int base = 1;
#ifdef local
	const int maxbase = 20;
#else
	const int maxbase = 20;
#endif
	u32 r[1 << maxbase];
	u32 dr[1 << maxbase];
	u32 _r[1 << maxbase];
//vector<u32> r = {0, 1};
//vector<u32> dr = {0, (1ull << 32) / mod};
	void prepare(int zeros) {
		assert(zeros <= maxbase);
		r[1] = 1;
		_r[1] = 1;
		dr[1] = (1ull << 32) / mod;
		if(base >= zeros) return;
//	r.resize(1 << zeros);
//	dr.resize(1 << zeros);
		while(base < zeros) {
			int z = power(G, (mod-1)>>base+1);
			int _z = power(z, mod - 2);
			for(int i = 1 << base - 1; i < 1 << base; i++) {
				r[i << 1] = r[i];
				_r[i << 1] = _r[i];
				dr[i << 1] = dr[i];
				r[i << 1 | 1] = (i64) r[i] * z % mod;
				_r[i << 1 | 1] = (i64) _r[i] * _z % mod;
				dr[i << 1 | 1] = (static_cast<i64>(r[i << 1 | 1]) << 32) / mod;
			}
			base++;
		}
	}
	const i64 low = ~0ull >> 32;
	const i64 high = low << 32;
	inline word load(i32 *A) {
		return _mm256_load_si256((word*) A);
	}
	inline word load(u32 *A) {
		return _mm256_load_si256((word *) A);
	}
	inline void store(word *ptr, word src) {
		_mm256_store_si256(ptr, src);
	}
	inline word wadd(word a, word b) {
		const word wmod = _mm256_set1_epi32(mod);
		a = _mm256_add_epi32(a, b);
		return _mm256_min_epu32(a, _mm256_sub_epi32(a, wmod));
	}
	inline word wsub(word a, word b) {
		const word wmod = _mm256_set1_epi32(mod);
		a = _mm256_sub_epi32(a, b);
		return _mm256_min_epu32(a, _mm256_add_epi32(a, wmod));
	}
	inline word wmul(word vv, word vc, word vd) {
		const word wmod = _mm256_set1_epi32(mod);
//	const word dwmod = _mm256_set1_epi64x(mod);
		const word dwlow = _mm256_set1_epi64x(low);
//	const word dwhigh = _mm256_set1_epi64x(high);
		word vv0 = _mm256_and_si256(vv, dwlow), vv1 = _mm256_srli_epi64(vv, 32);
		word vc0 = _mm256_and_si256(vc, dwlow), vc1 = _mm256_srli_epi64(vc, 32);
		word vd0 = _mm256_and_si256(vd, dwlow), vd1 = _mm256_srli_epi64(vd, 32);
 
		word vx0 = _mm256_mul_epu32(vv0, vc0);
		word vx1 = _mm256_mul_epu32(vv1, vc1);
		word vy0 = _mm256_mul_epu32(_mm256_srli_epi64(_mm256_mul_epu32(vv0, vd0), 32), wmod);
		word vy1 = _mm256_mul_epu32(_mm256_srli_epi64(_mm256_mul_epu32(vv1, vd1), 32), wmod);
		word vz0 = _mm256_sub_epi64(vx0, vy0), vz1 = _mm256_sub_epi64(vx1, vy1);
		word vz = _mm256_or_si256(_mm256_slli_epi64(vz1, 32), vz0);
		return wsub(vz, wmod);
	}
	inline void parallel0(i32 *A, i32 *B, u32 *C, u32 *D) {
		word va = load(A);
		word vb = load(B);
		_mm256_store_si256((word*)A, wadd(va, vb));
		_mm256_store_si256((word*)B, wmul(wsub(va, vb),load(C),load(D)));
	}
	inline void U0(int &va, int &vb, const u32 &vc, const u32 &vd) {
		int x = add(va, vb), y = sub(va, vb);
		int z = (i64) y * vc - ((i64) y * vd >> 32) * mod;
		va = x, vb = sub(z, mod);
	}
	void dft(int *a, int n) {
		assert((n & (n - 1)) == 0);
		int zeros = __builtin_ctz(n);
		prepare(zeros);
//	u32 *xr = r.data(), *xdr = dr.data();
		for(int i = n >> 1; i >= 8; i >>= 1) {
			for(int j = 0; j < n; j += i * 2) {
				for(int k = 0; k < i; k += 8) {
					parallel0(&a[j + k], &a[j + k + i],
					          &r[i + k], &dr[i + k]);
				}
			}
		}
		if(n >= 8) {
			for(int j = 0; j < n; j += 8) {
				U0(a[j + 0], a[j + 4], r[4], dr[4]);
				U0(a[j + 1], a[j + 5], r[5], dr[5]);
				U0(a[j + 2], a[j + 6], r[6], dr[6]);
				U0(a[j + 3], a[j + 7], r[7], dr[7]);
			}
		}
		if(n >= 4) {
			for(int j = 0; j < n; j += 4) {
				U0(a[j + 0], a[j + 2], r[2], dr[2]);
				U0(a[j + 1], a[j + 3], r[3], dr[3]);
			}
		}
		if(n >= 2) {
			for(int j = 0; j < n; j += 2) {
				U0(a[j], a[j + 1], r[1], dr[1]);
			}
		}
#undef U
	}
	inline void parallel1(i32 *A, i32 *B, u32 *C, u32 *D) {
		word va = load(A);
		word vb = wmul(load(B), load(C), load(D));
		_mm256_store_si256((word*)A, wadd(va, vb));
		_mm256_store_si256((word*)B, wsub(va, vb));
	}
	inline void U1(int &va, int &vb, const u32 &vc, const u32 &vd) {
		int x = va, y = (i64) vb * vc - ((i64) vb * vd >> 32) * mod;
		y = sub(y, mod);
		va = add(x, y), vb = sub(x, y);
	}
	void idft(i32 *a, int n) {
		assert((n & (n - 1)) == 0);
		int zeros = __builtin_ctz(n);
		prepare(zeros);
		if(n >= 2) {
			for(int i = 0; i < n; i += 2) {
				U1(a[i], a[i + 1], r[1], dr[1]);
			}
		}
		if(n >= 4) {
			for(int i = 0; i < n; i += 4) {
				U1(a[i + 0], a[i + 2], r[2], dr[2]);
				U1(a[i + 1], a[i + 3], r[3], dr[3]);
			}
		}
		if(n >= 8) {
			for(int i = 0; i < n; i += 8) {
				U1(a[i + 0], a[i + 4], r[4], dr[4]);
				U1(a[i + 1], a[i + 5], r[5], dr[5]);
				U1(a[i + 2], a[i + 6], r[6], dr[6]);
				U1(a[i + 3], a[i + 7], r[7], dr[7]);
			}
		}
		for(int i = 8; i < n; i <<= 1) {
			for(int j = 0; j < n; j += i * 2) {
				for(int k = 0; k < i; k += 8) {
					parallel1(&a[j + k], &a[j + k + i],
					          &r[i + k], &dr[i + k]);
				}
			}
		}
		reverse(a + 1, a + n);
		for(int i = 0; i < n; i++) {
			a[i] = (a[i] + (i64) (-a[i] & (n - 1)) * mod) >> zeros;
		}
	}
	int fa[1 << maxbase];
	void dft(vector<int> &a, int sz = -1) {
		if(sz == -1) sz = a.size();
		copy(a.data(), a.data() + sz, fa);
		dft(fa, sz);
		copy(fa, fa + sz, a.data());
	}
	void idft(vector<int> &a, int sz = -1) {
		if(sz == -1) sz = a.size();
		copy(a.data(), a.data() + sz, fa);
		idft(fa, sz);
		copy(fa, fa + sz, a.data());
	}
	const int H = 64;
	vector<int> multiply(vector<int> a, vector<int> b) {
		int need = a.size() + b.size() - 1;
		if(min(a.size(), b.size()) < H) {
			const u64 maxval = 14ull << 60;
			vector<u64> buf(need);
			for(int i = 0; i < (int) a.size(); i++) {
				for(int j = 0; j < (int) b.size(); j++) {
					buf[i + j] += (u64) a[i] * b[j];
					if(buf[i + j] >= maxval) {
						buf[i + j] %= mod;
					}
				}
			}
			vector<int> c(need);
			for(int i = 0; i < need; i++) c[i] = buf[i] % mod;
			return c;
		}
		int sz = need > 1 ? 1 << (32 - __builtin_clz(need - 1)) : 1;
		bool eq = a == b;
		a.resize(sz);
		b.resize(sz);
		dft(a, sz);
		if(eq) {
			copy(a.begin(), a.begin() + sz, b.begin());
		} else {
			dft(b, sz);
		}
		for(int i = 0; i < sz; i++) {
			a[i] = (i64) a[i] * b[i] % mod;
		}
		idft(a, sz);
		a.resize(need);
		return a;
	}
}
using namespace NTT;
