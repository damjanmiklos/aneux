// Sizing-field half-edge collapse + Delaunay flips (see coarsen.py).
//
// The coarse vertex set is a subset of the input one, so the levels nest, and
// a vertex dies only by collapsing into a neighbour.  Plain C ABI so ctypes can
// load it from any Python version (hemomesh 3.11 and aneurysmgnn 3.14 alike).
// Every step mirrors `_coarsen_python` in coarsen.py, which is the reference.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define CO_EXPORT extern "C" __declspec(dllexport)
#else
#define CO_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct V3 {
    double x, y, z;
};
inline V3 sub(const V3& a, const V3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 add(const V3& a, const V3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const V3& a) { return std::sqrt(dot(a, a)); }
inline V3 scale(const V3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }

struct Mesh {
    std::vector<V3> V;
    std::vector<std::array<int64_t, 3>> F;
    std::vector<std::vector<int64_t>> vf;   // vertex -> incident live faces
    std::vector<char> alive_f, alive_v, bnd;
    std::vector<int64_t> loop;
    std::vector<int64_t> rim_count;
    std::vector<double> h;
    std::vector<V3> N0;                     // unit normals of the original surface

    V3 fn(const std::array<int64_t, 3>& t) const {
        return cross(sub(V[t[1]], V[t[0]]), sub(V[t[2]], V[t[0]]));
    }
    double q(const std::array<int64_t, 3>& t) const {
        const V3 &a = V[t[0]], &b = V[t[1]], &c = V[t[2]];
        double area2 = norm(cross(sub(b, a), sub(c, a)));
        V3 e0 = sub(b, a), e1 = sub(c, b), e2 = sub(a, c);
        double s = dot(e0, e0) + dot(e1, e1) + dot(e2, e2);
        return 2.0 * std::sqrt(3.0) * area2 / std::max(s, 1e-30);
    }
    double hedge(int64_t a, int64_t b) const { return 0.5 * (h[a] + h[b]); }
    // the upper bound uses the finer end: R_template jumps at a junction, and
    // the mean would let a thin branch's first ring take the parent's size
    double hmax_edge(int64_t a, int64_t b) const { return std::min(h[a], h[b]); }
    double len(int64_t a, int64_t b) const { return norm(sub(V[a], V[b])); }

    std::vector<int64_t> nbrs(int64_t v) const {
        std::vector<int64_t> s;
        s.reserve(vf[v].size() * 2);
        for (int64_t f : vf[v])
            for (int k = 0; k < 3; ++k)
                if (F[f][k] != v) s.push_back(F[f][k]);
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        return s;
    }
    std::vector<int64_t> common_faces(int64_t a, int64_t b) const {
        std::vector<int64_t> out;
        for (int64_t f : vf[a])
            if (std::find(vf[b].begin(), vf[b].end(), f) != vf[b].end()) out.push_back(f);
        return out;
    }
    static bool contains(const std::vector<int64_t>& s, int64_t x) {
        return std::binary_search(s.begin(), s.end(), x);
    }
    static void erase_val(std::vector<int64_t>& s, int64_t x) {
        auto it = std::find(s.begin(), s.end(), x);
        if (it != s.end()) {
            *it = s.back();
            s.pop_back();
        }
    }
};

struct Params {
    double alpha, beta, cos_turn, q_min, cos_flip_new, cos_flip_dihedral, cos_anchor;
    int64_t min_rim;
};

// Remove b into a.  Returns false when the collapse would break the mesh.
bool try_collapse(const Mesh& M, const Params& P, int64_t b, int64_t a, double* worst_q) {
    std::vector<int64_t> fab = M.common_faces(a, b);
    if (fab.size() != 1 && fab.size() != 2) return false;
    if (M.bnd[b]) {
        if (!(M.bnd[a] && fab.size() == 1)) return false;
        if (M.rim_count[M.loop[b]] <= P.min_rim) return false;
    } else if (fab.size() == 1) {
        return false;
    }
    std::vector<int64_t> Na = M.nbrs(a), Nb = M.nbrs(b);
    std::vector<int64_t> opp;
    for (int64_t f : fab)
        for (int k = 0; k < 3; ++k)
            if (M.F[f][k] != a && M.F[f][k] != b) opp.push_back(M.F[f][k]);
    std::sort(opp.begin(), opp.end());
    opp.erase(std::unique(opp.begin(), opp.end()), opp.end());
    std::vector<int64_t> common;
    std::set_intersection(Na.begin(), Na.end(), Nb.begin(), Nb.end(), std::back_inserter(common));
    if (common != opp) return false;   // link condition
    if (!M.bnd[a] && !M.bnd[b]) {
        std::vector<int64_t> uni;
        std::set_union(Na.begin(), Na.end(), Nb.begin(), Nb.end(), std::back_inserter(uni));
        if ((int64_t)uni.size() - 2 < 3) return false;
    }
    for (int64_t c : Nb) {
        if (c == a) continue;
        if (M.len(a, c) > P.beta * M.hmax_edge(a, c)) return false;
    }
    double worst = 1.0;
    for (int64_t f : M.vf[b]) {
        if (std::find(fab.begin(), fab.end(), f) != fab.end()) continue;
        std::array<int64_t, 3> tri = M.F[f];
        V3 n0 = M.fn(tri);
        for (int k = 0; k < 3; ++k)
            if (tri[k] == b) tri[k] = a;
        V3 n1 = M.fn(tri);
        double l0 = norm(n0), l1 = norm(n1);
        if (l1 < 1e-14 || dot(n0, n1) < P.cos_turn * l0 * l1) return false;
        // the turn above is per collapse and can add up over many of them,
        // so every new triangle must also still face the original surface
        for (int k = 0; k < 3; ++k)
            if (dot(n1, M.N0[tri[k]]) < P.cos_anchor * l1) return false;
        double qn = M.q(tri);
        if (qn < P.q_min && qn < M.q(M.F[f])) return false;
        worst = std::min(worst, qn);
    }
    *worst_q = worst;
    return true;
}

void do_collapse(Mesh& M, int64_t b, int64_t a) {
    std::vector<int64_t> fab = M.common_faces(a, b);
    for (int64_t f : fab) {
        M.alive_f[f] = 0;
        for (int k = 0; k < 3; ++k) Mesh::erase_val(M.vf[M.F[f][k]], f);
    }
    for (int64_t f : M.vf[b]) {
        for (int k = 0; k < 3; ++k)
            if (M.F[f][k] == b) M.F[f][k] = a;
        M.vf[a].push_back(f);
    }
    M.vf[b].clear();
    M.alive_v[b] = 0;
    if (M.bnd[b]) M.rim_count[M.loop[b]] -= 1;
}

typedef std::tuple<double, int64_t, int64_t> Entry;

int64_t collapse_pass(Mesh& M, const Params& P) {
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    auto push = [&](int64_t a, int64_t b) {
        double r = M.len(a, b) / M.hedge(a, b);
        if (r < P.alpha) heap.emplace(r, std::min(a, b), std::max(a, b));
    };
    const int64_t m = (int64_t)M.F.size();
    for (int64_t f = 0; f < m; ++f) {
        if (!M.alive_f[f]) continue;
        for (int k = 0; k < 3; ++k) {
            int64_t x = M.F[f][k], y = M.F[f][(k + 1) % 3];
            if (x < y) {
                push(x, y);
            } else {
                // a boundary edge is seen from one face only
                std::vector<int64_t> ef = M.common_faces(x, y);
                if (ef.size() == 1) push(y, x);
            }
        }
    }
    int64_t nc = 0;
    while (!heap.empty()) {
        Entry e = heap.top();
        heap.pop();
        double r = std::get<0>(e);
        int64_t a = std::get<1>(e), b = std::get<2>(e);
        if (!(M.alive_v[a] && M.alive_v[b])) continue;
        if (M.common_faces(a, b).empty()) continue;
        if (std::fabs(M.len(a, b) / M.hedge(a, b) - r) > 1e-12) continue;
        double q1 = 0.0, q2 = 0.0;
        bool ok1 = try_collapse(M, P, b, a, &q1);
        bool ok2 = try_collapse(M, P, a, b, &q2);
        if (!(ok1 || ok2)) continue;
        int64_t keep;
        if (ok1 && (!ok2 || q1 >= q2)) {
            do_collapse(M, b, a);
            keep = a;
        } else {
            do_collapse(M, a, b);
            keep = b;
        }
        ++nc;
        for (int64_t c : M.nbrs(keep)) push(keep, c);
    }
    return nc;
}

double angle_at(const Mesh& M, int64_t p, int64_t x, int64_t y) {
    V3 u = sub(M.V[x], M.V[p]), w = sub(M.V[y], M.V[p]);
    double c = dot(u, w) / (norm(u) * norm(w) + 1e-30);
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c);
}

int64_t flip_pass(Mesh& M, const Params& P) {
    const double pi = 3.14159265358979323846;
    int64_t nf = 0;
    const int64_t m = (int64_t)M.F.size();
    for (int it = 0; it < 4; ++it) {
        int64_t changed = 0;
        for (int64_t f1 = 0; f1 < m; ++f1) {
            if (!M.alive_f[f1]) continue;
            for (int k = 0; k < 3; ++k) {
                int64_t a = M.F[f1][k], b = M.F[f1][(k + 1) % 3], c = M.F[f1][(k + 2) % 3];
                std::vector<int64_t> ef = M.common_faces(a, b);
                if (ef.size() != 2) continue;
                int64_t f2 = ef[0] == f1 ? ef[1] : ef[0];
                int64_t d = -1;
                for (int j = 0; j < 3; ++j)
                    if (M.F[f2][j] != a && M.F[f2][j] != b) d = M.F[f2][j];
                if (d < 0) continue;
                std::vector<int64_t> Nc = M.nbrs(c);
                if (Mesh::contains(Nc, d)) continue;
                if (M.len(c, d) > P.beta * M.hmax_edge(c, d)) continue;
                if ((int64_t)M.vf[a].size() <= (M.bnd[a] ? 2 : 3)) continue;
                if ((int64_t)M.vf[b].size() <= (M.bnd[b] ? 2 : 3)) continue;
                if (angle_at(M, c, a, b) + angle_at(M, d, a, b) <= pi + 1e-6) continue;
                V3 n1 = M.fn(M.F[f1]), n2 = M.fn(M.F[f2]);
                std::array<int64_t, 3> t1 = {a, d, c}, t2 = {d, b, c};
                V3 m1 = M.fn(t1), m2 = M.fn(t2);
                V3 nn = add(scale(n1, 1.0 / norm(n1)), scale(n2, 1.0 / norm(n2)));
                double c1 = dot(m1, nn) / (norm(m1) * norm(nn) + 1e-30);
                double c2 = dot(m2, nn) / (norm(m2) * norm(nn) + 1e-30);
                if (std::min(c1, c2) < P.cos_flip_new) continue;
                if (dot(n1, n2) < P.cos_flip_dihedral * norm(n1) * norm(n2)) continue;
                bool anchored = true;
                for (int j = 0; j < 3; ++j) {
                    if (dot(m1, M.N0[t1[j]]) < P.cos_anchor * norm(m1)) anchored = false;
                    if (dot(m2, M.N0[t2[j]]) < P.cos_anchor * norm(m2)) anchored = false;
                }
                if (!anchored) continue;
                Mesh::erase_val(M.vf[a], f2);
                Mesh::erase_val(M.vf[b], f1);
                M.F[f1] = t1;
                M.F[f2] = t2;
                M.vf[c].push_back(f2);
                M.vf[d].push_back(f1);
                ++changed;
                break;
            }
        }
        nf += changed;
        if (!changed) break;
    }
    return nf;
}

}  // namespace

// Returns the number of kept vertices (>= 0) or a negative error code.
// keep_out needs n_v slots, faces_out n_f*3, stats_out 2*passes.
CO_EXPORT int64_t coarsen_mesh(int64_t n_v, const double* V, int64_t n_f, const int64_t* F,
                               const double* h, const double* N0, double alpha, double beta, double cos_turn,
                               double q_min, int64_t min_rim, int64_t passes, int64_t do_flip,
                               double cos_flip_new, double cos_flip_dihedral, double cos_anchor,
                               int64_t* keep_out, int64_t* faces_out, int64_t* n_faces_out,
                               int64_t* stats_out) {
    if (n_v <= 0 || n_f <= 0) return -1;
    Mesh M;
    M.V.resize(n_v);
    for (int64_t i = 0; i < n_v; ++i) M.V[i] = {V[3 * i], V[3 * i + 1], V[3 * i + 2]};
    M.h.assign(h, h + n_v);
    M.N0.resize(n_v);
    for (int64_t i = 0; i < n_v; ++i) M.N0[i] = {N0[3 * i], N0[3 * i + 1], N0[3 * i + 2]};
    M.F.resize(n_f);
    M.vf.assign(n_v, {});
    for (int64_t f = 0; f < n_f; ++f) {
        for (int k = 0; k < 3; ++k) {
            int64_t v = F[3 * f + k];
            if (v < 0 || v >= n_v) return -2;
            M.F[f][k] = v;
            M.vf[v].push_back(f);
        }
    }
    M.alive_f.assign(n_f, 1);
    M.alive_v.assign(n_v, 1);
    M.bnd.assign(n_v, 0);
    M.loop.assign(n_v, -1);
    // boundary edges = edges used by exactly one face
    std::map<std::pair<int64_t, int64_t>, int> ecount;
    for (int64_t f = 0; f < n_f; ++f)
        for (int k = 0; k < 3; ++k) {
            int64_t x = M.F[f][k], y = M.F[f][(k + 1) % 3];
            ecount[{std::min(x, y), std::max(x, y)}] += 1;
        }
    std::vector<std::vector<int64_t>> badj(n_v);
    for (const auto& kv : ecount) {
        if (kv.second > 2) return -3;   // non-manifold input
        if (kv.second == 1) {
            int64_t x = kv.first.first, y = kv.first.second;
            M.bnd[x] = M.bnd[y] = 1;
            badj[x].push_back(y);
            badj[y].push_back(x);
        }
    }
    for (int64_t v0 = 0; v0 < n_v; ++v0) {
        if (!M.bnd[v0] || M.loop[v0] >= 0) continue;
        int64_t lid = (int64_t)M.rim_count.size();
        int64_t cnt = 0;
        std::vector<int64_t> stack = {v0};
        while (!stack.empty()) {
            int64_t v = stack.back();
            stack.pop_back();
            if (M.loop[v] >= 0) continue;
            M.loop[v] = lid;
            ++cnt;
            for (int64_t w : badj[v]) stack.push_back(w);
        }
        M.rim_count.push_back(cnt);
    }
    Params P{alpha, beta, cos_turn, q_min, cos_flip_new, cos_flip_dihedral, cos_anchor, min_rim};
    for (int64_t p = 0; p < passes; ++p) {
        int64_t nc = collapse_pass(M, P);
        int64_t nfl = do_flip ? flip_pass(M, P) : 0;
        stats_out[2 * p] = nc;
        stats_out[2 * p + 1] = nfl;
        if (nc == 0 && nfl == 0) {
            for (int64_t q = p + 1; q < passes; ++q) stats_out[2 * q] = stats_out[2 * q + 1] = -1;
            break;
        }
    }
    std::vector<int64_t> remap(n_v, -1);
    int64_t nk = 0;
    for (int64_t i = 0; i < n_v; ++i)
        if (M.alive_v[i]) {
            remap[i] = nk;
            keep_out[nk++] = i;
        }
    int64_t nfo = 0;
    for (int64_t f = 0; f < n_f; ++f) {
        if (!M.alive_f[f]) continue;
        for (int k = 0; k < 3; ++k) faces_out[3 * nfo + k] = remap[M.F[f][k]];
        ++nfo;
    }
    *n_faces_out = nfo;
    return nk;
}
