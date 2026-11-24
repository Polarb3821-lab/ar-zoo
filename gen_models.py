"""수학·과학 3D 모델(.glb)을 numpy만으로 생성 (단위: m)"""
import json, struct, itertools, os
import numpy as np

OUT = "/home/claude/ar-zoo/models"
os.makedirs(OUT, exist_ok=True)

# ---------- GLB writer ----------
def write_glb(path, parts):
    """parts: list of dict(pos, nrm, idx, color(rgba), metal, rough, double)"""
    bin_ = bytearray(); views = []; accs = []; meshes = []; mats = []; nodes = []
    def add(arr, target, comp, typ, minmax=False):
        nonlocal bin_
        while len(bin_) % 4: bin_ += b"\0"
        off = len(bin_); data = arr.tobytes(); bin_ += data
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        a = {"bufferView": len(views)-1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax: a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
        accs.append(a); return len(accs)-1
    for p in parts:
        pos = np.asarray(p["pos"], np.float32); nrm = np.asarray(p["nrm"], np.float32)
        idx = np.asarray(p["idx"], np.uint32).ravel()
        ap = add(pos, 34962, 5126, "VEC3", True); an = add(nrm, 34962, 5126, "VEC3")
        ai = add(idx, 34963, 5125, "SCALAR")
        mats.append({"pbrMetallicRoughness": {"baseColorFactor": list(p["color"]),
                     "metallicFactor": p.get("metal", 0.0), "roughnessFactor": p.get("rough", 0.5)},
                     "doubleSided": p.get("double", False)})
        meshes.append({"primitives": [{"attributes": {"POSITION": ap, "NORMAL": an},
                       "indices": ai, "material": len(mats)-1}]})
        nodes.append({"mesh": len(meshes)-1})
    while len(bin_) % 4: bin_ += b"\0"
    gltf = {"asset": {"version": "2.0", "generator": "ar-zoo"}, "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
            "materials": mats, "accessors": accs, "bufferViews": views,
            "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(bin_), 0x004E4942)); f.write(bin_)

def vertex_normals(pos, idx):
    n = np.zeros_like(pos)
    t = pos[idx]; fn = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    for k in range(3): np.add.at(n, idx[:, k], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l == 0] = 1
    return n / l

# ---------- primitives ----------
def param_surface(f, u0, u1, v0, v1, nu=120, nv=60):
    u = np.linspace(u0, u1, nu); v = np.linspace(v0, v1, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.stack(f(U, V), -1).reshape(-1, 3)
    idx = []
    for i in range(nu-1):
        for j in range(nv-1):
            a = i*nv+j; b = (i+1)*nv+j
            idx += [[a, b, a+1], [a+1, b, b+1]]
    idx = np.array(idx)
    return P, vertex_normals(P, idx), idx

def sphere(c, r, n=32):
    P, N, I = param_surface(lambda u, v: (r*np.sin(v)*np.cos(u), r*np.cos(v), r*np.sin(v)*np.sin(u)),
                            0, 2*np.pi, 1e-4, np.pi-1e-4, n, n//2)
    return P + np.asarray(c), N, I

def cylinder(a, b, r, n=20):
    a = np.asarray(a, float); b = np.asarray(b, float); d = b-a; L = np.linalg.norm(d); z = d/L
    x = np.cross(z, [0, 1, 0] if abs(z[1]) < .9 else [1, 0, 0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    th = np.linspace(0, 2*np.pi, n); P = []; N = []
    for t in (0, 1):
        for q in th:
            nn = np.cos(q)*x + np.sin(q)*y; P.append(a + t*d + r*nn); N.append(nn)
    I = []
    for i in range(n-1): I += [[i, n+i, i+1], [i+1, n+i, n+i+1]]
    return np.array(P), np.array(N), np.array(I)

def merge(items):
    P = []; N = []; I = []; o = 0
    for p, n, i in items: P.append(p); N.append(n); I.append(i+o); o += len(p)
    return np.vstack(P), np.vstack(N), np.vstack(I)

def part(geom, color, **kw):
    p, n, i = geom; return dict(pos=p, nrm=n, idx=i, color=color, **kw)

# ---------- convex polyhedra (flat shaded) ----------
def hull_faces(V):
    V = np.asarray(V, float); faces = {}
    for i, j, k in itertools.combinations(range(len(V)), 3):
        n = np.cross(V[j]-V[i], V[k]-V[i])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); d = V @ n - V[i] @ n
        if d.max() < 1e-6 or d.min() > -1e-6:
            if d.max() > 1e-6: n = -n
            on = tuple(sorted(np.where(abs(d) < 1e-6)[0]))
            faces[on] = n
    return faces

def polyhedron(V, edge_len):
    V = np.asarray(V, float)
    # 모서리 길이 맞추기
    dmin = min(np.linalg.norm(V[a]-V[b]) for a, b in itertools.combinations(range(len(V)), 2))
    V = V * (edge_len/dmin); V[:, 1] -= V[:, 1].min()
    P = []; N = []; I = []
    for ids, n in hull_faces(V).items():
        F = V[list(ids)]; c = F.mean(0)
        e1 = F[0]-c; e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        F = F[np.argsort(np.arctan2((F-c)@e2, (F-c)@e1))]
        o = len(P); P += list(F); N += [n]*len(F)
        for t in range(1, len(F)-1): I.append([o, o+t, o+t+1])
    return np.array(P), np.array(N), np.array(I)

phi = (1+5**.5)/2
TETRA = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
CUBE = list(itertools.product([-1, 1], repeat=3))
OCTA = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
ICO = [p for a in (-1, 1) for b in (-phi, phi) for p in ([0, a, b], [a, b, 0], [b, 0, a])]
DODE = CUBE + [p for a in (-1/phi, 1/phi) for b in (-phi, phi) for p in ([0, a, b], [a, b, 0], [b, 0, a])]

BLUE = (0.22, 0.47, 0.95, 1); TEAL = (0.1, 0.7, 0.65, 1); ORANGE = (0.98, 0.55, 0.15, 1)
PURPLE = (0.55, 0.35, 0.9, 1); PINK = (0.93, 0.35, 0.55, 1); GOLD = (0.95, 0.75, 0.2, 1)

models = {}
models["tetrahedron"]  = [part(polyhedron(TETRA, .30), ORANGE, rough=.35)]
models["cube"]         = [part(polyhedron(CUBE, .25), BLUE, rough=.35)]
models["octahedron"]   = [part(polyhedron(OCTA, .25), TEAL, rough=.35)]
models["dodecahedron"] = [part(polyhedron(DODE, .12), PURPLE, rough=.35)]
models["icosahedron"]  = [part(polyhedron(ICO, .15), PINK, rough=.35)]

S = .15  # 곡면 기준 크기 (m)
def lift(P, N, I, dy=None):
    P = P.copy(); P[:, 1] -= P[:, 1].min() if dy is None else dy; return P, N, I

models["torus"] = [part(lift(*param_surface(
    lambda u, v: ((2*S+S*np.cos(v))*np.cos(u), S*np.sin(v), (2*S+S*np.cos(v))*np.sin(u)),
    0, 2*np.pi, 0, 2*np.pi, 120, 48)), GOLD, metal=.6, rough=.3, double=True)]
models["mobius"] = [part(lift(*param_surface(
    lambda u, v: ((2*S+v*np.cos(u/2))*np.cos(u), v*np.sin(u/2)+S, (2*S+v*np.cos(u/2))*np.sin(u)),
    0, 2*np.pi, -S*.8, S*.8, 160, 16)), TEAL, double=True, rough=.4)]
models["saddle"] = [part(lift(*param_surface(
    lambda u, v: (u, (u**2-v**2)/(2*S*1.5)+S, v), -2*S, 2*S, -2*S, 2*S, 60, 60)), PURPLE, double=True, rough=.4)]
models["helicoid"] = [part(lift(*param_surface(
    lambda u, v: (v*np.cos(u), u*S/np.pi*1.0, v*np.sin(u)), 0, 4*np.pi, -1.5*S, 1.5*S, 200, 20)),
    ORANGE, double=True, rough=.4)]
models["catenoid"] = [part(lift(*param_surface(
    lambda u, v: (S*np.cosh(v/S)*np.cos(u), v, S*np.cosh(v/S)*np.sin(u)), 0, 2*np.pi, -1.4*S, 1.4*S, 100, 50)),
    BLUE, double=True, rough=.4)]
def klein(u, v):  # 클라인 병 (병 모양 몰입)
    r = 4*(1-np.cos(u)/2)
    x = np.where(u < np.pi, 6*np.cos(u)*(1+np.sin(u)) + r*np.cos(u)*np.cos(v), 6*np.cos(u)*(1+np.sin(u)) + r*np.cos(v+np.pi))
    y = np.where(u < np.pi, 16*np.sin(u) + r*np.sin(u)*np.cos(v), 16*np.sin(u))
    z = r*np.sin(v); k = .018
    return x*k, -y*k, z*k
models["klein"] = [part(lift(*param_surface(klein, 0, 2*np.pi, 0, 2*np.pi, 160, 48)),
                   (0.2, 0.65, 0.9, 0.85), double=True, rough=.25)]

# ---------- 과학 ----------
CPK = {"H": ((0.95, 0.95, 0.95, 1), .035), "O": ((0.9, 0.15, 0.15, 1), .06),
       "C": ((0.2, 0.2, 0.2, 1), .06), "N": ((0.2, 0.35, 0.95, 1), .058)}
BOND = (0.7, 0.7, 0.72, 1)
def molecule(atoms, bonds, lift_y=.12):
    out = []; atoms = [(e, np.asarray(p)+[0, lift_y, 0]) for e, p in atoms]
    by = {}
    for e, p in atoms: by.setdefault(e, []).append(sphere(p, CPK[e][1]))
    for e, g in by.items(): out.append(part(merge(g), CPK[e][0], rough=.3))
    out.append(part(merge([cylinder(atoms[a][1], atoms[b][1], .012) for a, b in bonds]), BOND, rough=.5))
    return out

L = .12; th = np.radians(104.5)/2
models["water"] = molecule([("O", [0, 0, 0]), ("H", [L*np.sin(th), -L*np.cos(th)*0+L*np.cos(th)*-1, 0]),
                            ("H", [-L*np.sin(th), -L*np.cos(th), 0])], [(0, 1), (0, 2)])
t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/np.sqrt(3)*.13
models["methane"] = molecule([("C", [0, 0, 0])]+[("H", v) for v in t], [(0, i) for i in range(1, 5)])
models["co2"] = molecule([("C", [0, 0, 0]), ("O", [.14, 0, 0]), ("O", [-.14, 0, 0])], [(0, 1), (0, 2)], .08)
models["ammonia"] = molecule([("N", [0, 0, 0])]+[("H", [.115*np.cos(a), -.045, .115*np.sin(a)])
                    for a in np.radians([90, 210, 330])], [(0, 1), (0, 2), (0, 3)])

# DNA 이중나선 (단순화): 10염기쌍/회전
R = .06; rise = .017; nbp = 30; base_cols = [(0.95, 0.3, 0.3, 1), (0.3, 0.6, 0.95, 1), (0.3, 0.8, 0.4, 1), (0.95, 0.8, 0.2, 1)]
bb1 = []; bb2 = []; rungs = {k: [] for k in range(4)}
for i in range(nbp):
    a = i*2*np.pi/10; y = i*rise + .02
    p1 = np.array([R*np.cos(a), y, R*np.sin(a)]); p2 = np.array([R*np.cos(a+2.4), y, R*np.sin(a+2.4)])
    bb1.append(sphere(p1, .011, 16)); bb2.append(sphere(p2, .011, 16))
    mid = (p1+p2)/2; k = i % 4
    rungs[k].append(cylinder(p1, mid, .005, 10)); rungs[[1, 0, 3, 2][k]].append(cylinder(mid, p2, .005, 10))
    if i:
        q = (i-1)*2*np.pi/10; yq = (i-1)*rise + .02
        bb1.append(cylinder([R*np.cos(q), yq, R*np.sin(q)], p1, .006, 10))
        bb2.append(cylinder([R*np.cos(q+2.4), yq, R*np.sin(q+2.4)], p2, .006, 10))
models["dna"] = [part(merge(bb1), (0.85, 0.85, 0.9, 1), metal=.3, rough=.3),
                 part(merge(bb2), (0.6, 0.6, 0.7, 1), metal=.3, rough=.3)] + \
                [part(merge(g), base_cols[k], rough=.4) for k, g in rungs.items()]

# 보어 원자 모형 (탄소: 전자 2, 4)
nuc = []; rng = np.random.default_rng(1)
for i in range(12):
    d = rng.normal(size=3); d = d/np.linalg.norm(d)*.022
    nuc.append((i % 2, sphere(d+[0, .2, 0], .016, 16)))
orbit = lambda r: lift(*param_surface(lambda u, v: ((r+.002*np.cos(v))*np.cos(u), .002*np.sin(v), (r+.002*np.cos(v))*np.sin(u)),
                                       0, 2*np.pi, 0, 2*np.pi, 120, 8), dy=-.2)
elec = [sphere([.08*np.cos(a), .2, .08*np.sin(a)], .009, 16) for a in (0, np.pi)] + \
       [sphere([.15*np.cos(a), .2, .15*np.sin(a)], .009, 16) for a in np.arange(4)*np.pi/2+.4]
models["atom"] = [part(merge([g for k, g in nuc if k == 0]), (0.9, 0.25, 0.25, 1), rough=.4),
                  part(merge([g for k, g in nuc if k == 1]), (0.75, 0.75, 0.8, 1), rough=.4),
                  part(merge([orbit(.08), orbit(.15)]), (0.6, 0.7, 0.9, 0.6), rough=.5),
                  part(merge(elec), (0.25, 0.55, 1, 1), metal=.2, rough=.2)]

for name, parts in models.items():
    write_glb(f"{OUT}/{name}.glb", parts)
    size = os.path.getsize(f"{OUT}/{name}.glb")
    allp = np.vstack([p["pos"] for p in parts]); ext = allp.max(0)-allp.min(0)
    print(f"{name:13s} {size/1024:7.1f} KB  size(m) {ext.round(2)}")
