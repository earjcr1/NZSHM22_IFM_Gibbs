#!/usr/bin/env python3
"""Each subsection's share of the gridded seismicity, from the rupture set's fault_sections.geojson alone, as NZSHM22's
OpenSHA code computes it for the sub-seismogenic MFDs (the user's request, 2026-11-23: MFD_GR from a rupture set made
without an inversion; nzshm-opensha's SectionPolygons, NZSHM22_FaultPolyMgr and GriddedSeisUtils.pdfValForSection).

    from fault_polygons import section_shares
    pdf_s = section_shares(RUPTURE_DIR)          # one per subsection; their sum is the on-fault fraction

The polygons (SectionPolygons, with NZSHM22's buffer 12 km and minimum 0):
  - each parent fault's trace (its subsections' traces joined, repeated points dropped) widened by a buffer of
    12 km x (dip - 50) / 40 on both sides along the dip direction (none for dips of 50 degrees or less);
  - cut into subsections by 100-km envelopes whose border bisects each subsection boundary along the dip direction;
  - each merged with its subsection's surface projection: the trace, and the trace moved by LowDepth / tan(dip) along
    the dip direction (the Stirling surface's perimeter);
  - with Java's Area conventions: paths read even-odd, and every cleaning fills holes.
The shares (NZSHM22_FaultPolyMgr), on the NZ_RECTANGLE grid (0.1 degrees from -50 / 165, 200 x 200 nodes; the north and
east edges fall outside by Java's insideness): a subsection's fraction of a cell is its overlap's area over the cell's
(SectionPolygons.getExtent: each path projected about its bounding box's centre); in each cell the fractions are
scaled to sum to the cell's covered fraction; a subsection's share is the sum over cells of the grid's PDF x its
scaled fraction. The PDF: NZSHM22's gridded seismicity, FLOOR_ADDOPTIEEPASCOMB_CRU ('hybridEEPAS', the spatial PDF of
the 19k, 211k and 630k solutions; data/Floor_AddoptiEEPAScomb-CRU_pdf.csv, from nzshm-opensha's resources), normalised
over the grid.

Checked against the solutions at 19,000, 211,000 and 630,000 ruptures: the polygons match sect_polygons.geojson to
~1e-6 of their areas; 77-88% of the shares match sub_seismo_on_fault_mfds.csv exactly, the sum is 0.04-0.11% high and
some subsections up to 20% high. The difference is a quirk of Java's geometry: subtracting adjacent polygons from a cell
can leave a zero-width spike, and SectionPolygons.cleanBorder, which drops every repeated vertex of a path, then cuts a
corner, so NZSHM22's covered fraction comes out below the geometry's in 3-5% of the shared cells. NZSHM22's own code
(nzshm-opensha's jar) reproduces the stored associations exactly; the user chose pure Python (2026-11-23). Through
the partitioning, the supra-seismogenic target is within 0.15% (19k), 0.075% (211k) and 0.046% (630k) of NZSHM22's.
"""
import csv
import json
import os

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

GRID_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                        'Floor_AddoptiEEPAScomb-CRU_pdf.csv')
R_EARTH = 6371.0072                                     # OpenSHA's GeoTools.EARTH_RADIUS_MEAN, km
D2R = np.pi / 180
BUFFER, MIN_BUFFER, MAX_BUFFER_DIP = 12.0, 0.0, 50.0    # NZSHM22_FaultPolyParameters; SectionPolygons.MAX_BUF_DIP
ENVELOPE = 100.0                                         # SectionPolygons.BUF, km
SAME = 1e-9                                              # SectionPolygons.TOL, degrees
LAT0, LON0, STEP, NLAT, NLON = -50.0, 165.0, 0.1, 200, 200


# ---- OpenSHA's LocationUtils, on a sphere ---------------------------------------------------------------------------
def _dest(lat, lon, az, dist):
    """LocationUtils.location: the point dist km from (lat, lon) along azimuth az (degrees)."""
    la, lo, a = lat * D2R, lon * D2R, az * D2R
    ad = dist / R_EARTH
    la2 = np.arcsin(np.sin(la) * np.cos(ad) + np.cos(la) * np.sin(ad) * np.cos(a))
    lo2 = lo + np.arctan2(np.sin(a) * np.sin(ad) * np.cos(la), np.cos(ad) - np.sin(la) * np.sin(la2))
    return la2 / D2R, lo2 / D2R


def _azimuth(lat1, lon1, lat2, lon2):
    la1, la2, dlo = lat1 * D2R, lat2 * D2R, (lon2 - lon1) * D2R
    a = np.arctan2(np.sin(dlo) * np.cos(la2), np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo))
    return ((a + 2 * np.pi) % (2 * np.pi)) / D2R


def horz_km(lat1, lon1, lat2, lon2):
    """LocationUtils.horzDistance (haversine)."""
    la1, la2 = lat1 * D2R, lat2 * D2R
    c = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lon2 - lon1) * D2R / 2) ** 2
    return R_EARTH * 2 * np.arctan2(np.sqrt(c), np.sqrt(1 - c))


# ---- java.awt.geom.Area's conventions ---------------------------------------------------------------------------------
def _parts(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    return [q for h in getattr(g, 'geoms', []) for q in _parts(h)]


def _union(parts):
    parts = [p for p in parts if p is not None and not p.is_empty and p.area > 0]
    return unary_union(parts) if parts else Polygon()


def _even_odd(xy):
    """new Area(path): the closed path (lon, lat rows) read even-odd."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return Polygon()
    if not np.array_equal(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[:1]])
    p = Polygon(xy)
    if p.is_valid:
        return p if p.area > 0 else Polygon()
    x0, y0, x1, y1 = xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1]
    keep = []
    for f in polygonize(unary_union(LineString(xy))):          # the faces; a face is inside if a ray crosses the path oddly
        q = f.representative_point()
        cross = ((y0 > q.y) != (y1 > q.y)) & (q.x < (x1 - x0) * (q.y - y0) / np.where(y1 != y0, y1 - y0, 1) + x0)
        if np.sum(cross) % 2:
            keep.append(f)
    return _union(keep)


def _paths(g):
    """SectionPolygons.areaToLocLists: every path, outer borders and holes alike."""
    return [r for p in _parts(g) for r in [p.exterior] + list(p.interiors)]


def _clean(g):
    """SectionPolygons.cleanBorder: each path added back as a solid area, so holes fill."""
    return _union([_even_odd(np.asarray(r.coords)) for r in _paths(g) if len(r.coords) >= 4])


def _singular(g):
    return g is None or g.is_empty or (g.geom_type == 'Polygon' and not g.interiors)


def _hard_merge(g):
    """SectionPolygons.hardMerge: two parts merged only when a shift of 1e-9 degrees makes them one."""
    ps = [_even_odd(np.asarray(r.coords)) for r in _paths(g)]
    if len(ps) != 2 or ps[0].distance(ps[1]) > 4 * SAME:
        return None
    m = unary_union([ps[0], ps[1].buffer(SAME, join_style=2)])
    return m if _singular(m) else None


def extent_km2(g):
    """SectionPolygons.getExtent: each path's area, projected about its bounding box's centre, summed."""
    tot = 0.0
    for ring in _paths(g):
        xy = np.asarray(ring.coords)
        minx, miny, maxx, maxy = ring.bounds
        la0, lo0 = (miny + maxy) / 2, (minx + maxx) / 2
        az = _azimuth(la0, lo0, xy[:, 1], xy[:, 0]) * D2R
        d = horz_km(la0, lo0, xy[:, 1], xy[:, 0])
        x, y = np.sin(az) * d, np.cos(az) * d
        tot += 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
    return tot


# ---- SectionPolygons ------------------------------------------------------------------------------------------------
def section_polygons(feats):
    """{subsection: polygon or None}, as SectionPolygons.create(subsections, 12, null, 0); feats: fault_sections.geojson's
    features, in subsection order."""
    tr = [np.asarray(f['geometry']['coordinates'], float)[:, :2] for f in feats]      # (lon, lat)
    P = [f['properties'] for f in feats]
    by_parent = {}
    for i, p in enumerate(P):
        by_parent.setdefault(p['ParentID'], []).append(i)
    poly = {}
    for subs in by_parent.values():
        pts = []                                            # the parent's trace, repeated points dropped (removeDupes)
        for i in subs:
            for x, y in tr[i]:
                if not any(abs(x - a) <= SAME and abs(y - b) <= SAME for a, b in pts):
                    pts.append((x, y))
        dip, ddir = P[subs[0]]['DipDeg'], P[subs[0]]['DipDir']
        buf = max(MIN_BUFFER, (dip - MAX_BUFFER_DIP) / (90 - MAX_BUFFER_DIP) * BUFFER)
        if buf <= 0:                                        # no zone polygon: the subsections keep their surfaces alone
            continue
        back = (ddir + 180) % 360
        pieces = []
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):   # buildBufferPoly
            a1, b1 = _dest(ay, ax, ddir, buf), _dest(by, bx, ddir, buf)
            b2, a2 = _dest(by, bx, back, buf), _dest(ay, ax, back, buf)
            pieces.append(_even_odd([(ax, ay), (a1[1], a1[0]), (b1[1], b1[0]), (bx, by), (b2[1], b2[0]), (a2[1], a2[0])]))
        rest = _clean(_union(pieces))
        if len(subs) == 1:
            poly[subs[0]] = rest
            continue
        for k, i in enumerate(subs):                        # initPolys: the parent sliced subsection by subsection
            on_trace = lambda part, t=tr[i]: any(part.intersects(Point(x, y)) for x, y in t)
            if k == len(subs) - 1:
                if _singular(rest):
                    poly[i] = rest
                else:
                    for r in _paths(rest):
                        part = _even_odd(np.asarray(r.coords))
                        if on_trace(part):
                            poly[i] = part
                        else:
                            j = subs[k - 1]
                            prev = _clean(_union([poly.get(j), part]))
                            poly[j] = prev if _singular(prev) else _hard_merge(prev)
                break
            (x1, y1), (x2, y2) = tr[i][-2], tr[i][-1]       # createSubSecEnvelope
            az_back = _azimuth(y2, x2, y1, x1)
            dd = P[i]['DipDir']
            u, u2 = _dest(y2, x2, dd, ENVELOPE), _dest(y2, x2, (dd + 180) % 360, ENVELOPE)
            ua, ub = _dest(u[0], u[1], az_back, ENVELOPE), _dest(u2[0], u2[1], az_back, ENVELOPE)
            env = _even_odd([(ua[1], ua[0]), (u[1], u[0]), (x2, y2), (u2[1], u2[0]), (ub[1], ub[0])])
            sub = _union(_parts(rest.intersection(env)))
            if sub.is_empty:
                poly[i] = None
                continue
            sub = _clean(sub)
            left = None
            if _singular(sub):
                poly[i] = sub
            else:
                for r in _paths(sub):
                    part = _even_odd(np.asarray(r.coords))
                    if on_trace(part):
                        poly[i] = part
                    else:
                        left = part
            rest = _clean(_union(_parts(rest.difference(env))))
            if left is not None:                            # a piece off the trace goes back to the parent, or else to
                merged = _clean(_union([rest, left]))       # the subsection before
                if not _singular(merged):
                    merged = _hard_merge(merged)
                if merged is None:
                    j = subs[k - 1]
                    prev = _clean(_union([poly.get(j), left]))
                    poly[j] = prev if _singular(prev) else _hard_merge(prev)
                else:
                    rest = merged
    out = {}
    for i, p in enumerate(P):                               # cleanPolys, then mergeDownDip
        z = poly.get(i)
        z = None if z is None or z.is_empty else _clean(z)
        tan = np.tan(p['DipDeg'] * D2R)
        top = [_dest(y, x, p['DipDir'], p['UpDepth'] / tan) for x, y in tr[i]]
        bot = [_dest(y, x, p['DipDir'], p['LowDepth'] / tan) for x, y in tr[i]][::-1]
        surf = _even_odd([(b, a) for a, b in top + bot] + [(top[0][1], top[0][0])])
        m = _union([z, surf])
        out[i] = None if m.is_empty else (m if _singular(m) else _clean(m))
    return out


# ---- NZSHM22_FaultPolyMgr and pdfValForSection ------------------------------------------------------------------------
def _cell(i, j):
    lat, lon = round(LAT0 + STEP * i, 8), round(LON0 + STEP * j, 8)
    return box(lon - STEP / 2, lat - STEP / 2, lon + STEP / 2, lat + STEP / 2)


def grid_pdf(path=GRID_CSV):
    """The gridded seismicity on the 200 x 200 nodes (lat-major), normalised over them (NZSHM22_GriddedData:
    locations keyed by Math.round(x * 10))."""
    key = lambda x: int(np.floor(x * 10 + 0.5))
    tab = {}
    with open(path) as f:
        for r in csv.reader(f):
            try:
                tab[(key(float(r[0])), key(float(r[1])))] = float(r[2])
            except (ValueError, IndexError):
                pass
    pdf = np.array([tab.get((key(LAT0 + STEP * i), key(LON0 + STEP * j)), 0.0) for i in range(NLAT) for j in range(NLON)])
    if not pdf.sum() > 0:
        raise ValueError(f'{path}: no gridded seismicity inside the NZ rectangle')
    return pdf / pdf.sum()


def shares_from_polygons(polys, n_sect, pdf):
    """Each subsection's share: the sum over cells of pdf x its overlap fraction, scaled within each cell to the cell's
    covered fraction (NZSHM22_FaultPolyMgr.getScaledNodeFractions)."""
    frac = {}
    for s, P in polys.items():
        if P is None or P.is_empty:
            continue
        minx, miny, maxx, maxy = P.bounds
        for i in range(max(int(np.floor((miny - LAT0 + STEP / 2) / STEP)), 0), min(int(np.ceil((maxy - LAT0 + STEP / 2) / STEP)), NLAT)):
            for j in range(max(int(np.floor((minx - LON0 + STEP / 2) / STEP)), 0), min(int(np.ceil((maxx - LON0 + STEP / 2) / STEP)), NLON)):
                c = _cell(i, j)
                inter = _clean(_union(_parts(c.intersection(P))))
                if not inter.is_empty:
                    frac[(s, i * NLON + j)] = extent_km2(inter) / extent_km2(c)
    by_node = {}
    for (s, n), v in frac.items():
        by_node.setdefault(n, []).append((s, v))
    out = np.zeros(n_sect)
    for n, lst in by_node.items():
        c = _cell(*divmod(n, NLON))
        uncovered = _clean(_union(_parts(c.difference(unary_union([polys[s] for s, _ in lst])))))
        covered = 1.0 - extent_km2(uncovered) / extent_km2(c)
        tot = sum(v for _, v in lst)
        for s, v in lst:
            out[s] += pdf[n] * v / tot * covered
    return out


def section_shares(rupture_dir, grid_csv=GRID_CSV):
    """Each subsection's share of the gridded seismicity (fault_sections.geojson's order)."""
    feats = json.load(open(os.path.join(rupture_dir, 'fault_sections.geojson')))['features']
    return shares_from_polygons(section_polygons(feats), len(feats), grid_pdf(grid_csv))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('rupture_dir')
    ap.add_argument('--grid', default=GRID_CSV)
    a = ap.parse_args()
    sh = section_shares(a.rupture_dir, a.grid)
    print(f'{len(sh)} subsections; on-fault fraction {sh.sum():.6f}; {np.sum(sh > 0)} with a share')
