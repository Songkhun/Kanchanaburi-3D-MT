"""Reproducible CPU-rendered animation of the supplied WSINV3DMT model.

Run from anywhere with mtviewer/.venv/bin/python. No OpenGL context is needed.
The reader, axis order and centred mesh follow matlab/read_invertedmodel.m.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/kan-mpl')
sys.path.insert(0, str(ROOT / 'mtviewer/src'))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
from matplotlib import colormaps
import pyvista as pv
import imageio_ffmpeg
from mtviewer.io.wsinv3dmt import read_ws_model, read_ws_data
from mtviewer.render3d.scene import model_grid, isosurface

OUT = Path(__file__).resolve().parent
W, H, FPS, DURATION = 1920, 1080, 30, 90
BG = (247, 249, 251)
INK = (24, 43, 61)
MUTED = (82, 99, 112)
ACCENT = (25, 116, 141)
CMAP = colormaps['RdYlBu']  # low resistivity warm; high resistivity cool
FONTDIR = Path('/System/Library/Fonts/Supplemental')


def font(size, bold=False):
    return ImageFont.truetype(str(FONTDIR / ('Arial Bold.ttf' if bold else 'Arial.ttf')), size)


FONTS = {s: font(s) for s in (18, 20, 22, 24, 26, 28, 30, 34, 42)}
BOLD = {s: font(s, True) for s in (20, 24, 26, 28, 30, 34, 42, 48)}


def smooth(q):
    q = np.clip(q, 0, 1)
    return q*q*(3-2*q)


def rgb(logrho):
    return np.round(np.asarray(CMAP(np.clip((np.asarray(logrho)+1)/5, 0, 1)))[..., :3]*255).astype(np.uint8)


model = read_ws_model(ROOT / 'modPAPER/kansea4_model.01_01')
response = read_ws_data(ROOT / 'modPAPER/kansea4_resp.01_01')
stations = np.column_stack([response.site_xy[:, 1]/1000, response.site_xy[:, 0]/1000,
                            np.full(39, 0.12)])
old_sites = loadmat(ROOT / 'matlab/site3d.mat')
station_error = float(np.max(np.abs(response.site_xy - np.column_stack(
    [old_sites['xss'].ravel(), old_sites['yss'].ravel()])*1000)))
assert station_error < 10, 'Surface overlay and response station frames do not match'

# Retain the original cell values and exact boundaries within the viewing window.
ni = np.flatnonzero((model.x_edges[:-1] >= -80000) & (model.x_edges[1:] <= 80000))
ei = np.flatnonzero((model.y_edges[:-1] >= -80000) & (model.y_edges[1:] <= 80000))
di = np.flatnonzero(model.z_edges[1:] <= 36000)
N = model.x_edges[np.r_[ni, ni[-1]+1]]/1000
E = model.y_edges[np.r_[ei, ei[-1]+1]]/1000
D = model.z_edges[:len(di)+1]/1000
LOG = np.log10(model.rho[np.ix_(ni, ei, di)])


def plane_z(depth):
    iz = int(np.clip(np.searchsorted(D, depth, side='right')-1, 0, len(D)-2))
    ee, nn = np.meshgrid(E[:-1], N[:-1])
    ex, ny = np.meshgrid(E[1:], N[1:])
    v = np.stack([np.stack([ee, nn], -1), np.stack([ex, nn], -1),
                  np.stack([ex, ny], -1), np.stack([ee, ny], -1)], -2).reshape(-1, 4, 2)
    v = np.concatenate([v, np.full((*v.shape[:2], 1), -depth)], -1)
    return v, rgb(LOG[:, :, iz]).reshape(-1, 3)


def plane_e(east):
    ie = int(np.clip(np.searchsorted(E, east, side='right')-1, 0, len(E)-2))
    nn, dd = np.meshgrid(N[:-1], -D[:-1], indexing='ij')
    ny, dz = np.meshgrid(N[1:], -D[1:], indexing='ij')
    v = np.stack([np.stack([nn, dd], -1), np.stack([ny, dd], -1),
                  np.stack([ny, dz], -1), np.stack([nn, dz], -1)], -2).reshape(-1, 4, 2)
    v = np.concatenate([np.full((*v.shape[:2], 1), east), v], -1)
    return v, rgb(LOG[:, ie, :]).reshape(-1, 3)


def plane_n(north):
    jn = int(np.clip(np.searchsorted(N, north, side='right')-1, 0, len(N)-2))
    ee, dd = np.meshgrid(E[:-1], -D[:-1], indexing='ij')
    ex, dz = np.meshgrid(E[1:], -D[1:], indexing='ij')
    v = np.stack([np.stack([ee, dd], -1), np.stack([ex, dd], -1),
                  np.stack([ex, dz], -1), np.stack([ee, dz], -1)], -2).reshape(-1, 4, 2)
    v = np.stack([v[..., 0], np.full(v.shape[:2], north), v[..., 1]], -1)
    return v, rgb(LOG[jn, :, :]).reshape(-1, 3)


# The original MATLAB isosurface example excludes the shallow 0–3.5 km interval.
g = model_grid(model, max_depth=36000).clip_box([-80, 80, -80, 80, -36, -3.5], invert=False)
iso = isosurface(g, 10).triangulate().compute_normals(cell_normals=True, point_normals=False)
ISO_V = np.asarray(iso.points)[np.asarray(iso.faces).reshape(-1, 4)[:, 1:]]
ISO_NORMAL = np.asarray(iso.cell_data['Normals'])
assert len(ISO_V) > 0

surface_v, surface_c = plane_z(0.001)
overlays = []
for filename, a, b, color, width, name in [
    ('KANSRIV', 'xxk', 'yyk', (142, 151, 159), 2, 'Province boundary'),
    ('KANSRIV', 'xxs', 'yys', (22, 116, 180), 3, 'Reservoir outlines'),
    ('KANSRIV', 'xxv', 'yyv', (22, 116, 180), 3, 'Reservoir outlines'),
    ('fault_epicenter', 'xxf', 'yyf', (179, 49, 100), 3, 'Fault traces'),
]:
    data = loadmat(ROOT / 'matlab' / (filename+'.mat'))
    pts = np.column_stack([data[b].ravel(), data[a].ravel(), np.full(data[a].size, 0.07)])
    # Split on NaNs, map boundaries and anomalously large jumps.
    ok = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, :2]) <= 80).all(axis=1)
    cuts = np.flatnonzero(~ok)
    for run in np.split(np.arange(len(pts)), cuts):
        run = run[ok[run]]
        if len(run) < 2:
            continue
        for part in np.split(run, np.flatnonzero(np.linalg.norm(np.diff(pts[run, :2], axis=0), axis=1)>8)+1):
            if len(part) >= 2:
                overlays.append((pts[part], color, width, name))


def camera(az, el):
    a, e = np.deg2rad([az, el])
    towards = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0])
    up = np.cross(towards, right)
    return np.stack([right, up, towards])


def project(points, basis, el):
    target = np.array([0, 0, -12*(1-smooth((el-30)/60))])
    p = (points-target) @ basis.T
    scale = 4.35 - 0.35*smooth((el-30)/60)
    return np.stack([700+p[..., 0]*scale, 530-p[..., 1]*scale], -1), p[..., 2]


def mix_color(color, amount):
    return tuple(np.round(np.array(color)*amount+np.array(BG)*(1-amount)).astype(int))


def section_state(t):
    az, el = -90., 90.
    planes = [('z', 0.001, 1.)]
    iso_alpha, overlay_alpha, box_alpha = 0., 0., 0.
    title, detail = 'The survey surface', '39 MT stations across the Kanchanaburi survey area'
    caption = 'Magnetotelluric measurements reveal variations in subsurface electrical resistivity.'
    depth, section = 0., None
    if t >= 8:
        q = smooth((t-8)/10)
        az, el = -90+30*q, 90-60*q
        overlay_alpha = smooth((t-8)/3)
        title, detail = 'Surface context', 'MT stations, fault traces and reservoir outlines'
        caption = 'Surface features provide geographical context for the subsurface model.'
    if t >= 18:
        q = smooth((t-18)/5)
        planes = [('z', .001, 1-q), ('e', 0., q), ('n', 0., q)]
        box_alpha = q
        title, detail = 'A three-dimensional resistivity model', '160 × 160 km viewing window · 0–36 km depth'
        caption = 'Intersecting sections reveal the three-dimensional distribution of resistivity.'
    if t >= 28:
        # Six readable holds joined by smooth movements, covering 18 seconds.
        depths = [.3, 1., 3.5, 9., 18.5, 35.999]
        phase = min((t-28)/3, 5.9999)
        j = min(int(phase), 5)
        prev = .001 if j == 0 else depths[j-1]
        depth = prev+(depths[j]-prev)*smooth((phase-j)/.55)
        planes = [('z', depth, 1.)]
        box_alpha = 1.
        title, detail = 'Descending through the crust', f'Horizontal slice · depth {depth:.2f} km'
        caption = 'Successive depth slices show how conductive and resistive regions vary beneath the surface.'
    if t >= 46:
        q = smooth((t-46)/8) if t < 54 else smooth((t-54)/8)
        direction = 'e' if t < 54 else 'n'
        value = -65+130*q
        planes = [(direction, value, 1.)]
        section = (direction, value)
        title = 'Vertical sections through the model'
        detail = (f'North–south section · east {value:+.1f} km' if direction == 'e'
                  else f'East–west section · north {value:+.1f} km')
        caption = 'Moving cross-sections reveal the depth extent and geometry of resistivity contrasts.'
    if t >= 62:
        q = smooth((t-62)/14)
        az = -60+120*q
        iso_alpha = smooth((t-62)/3)
        planes = [('e', 0., .18*(1-smooth((t-62)/4)))]
        section = None
        title, detail = 'The geometry of low-resistivity regions', '10 Ω·m isosurface · depth interval 3.5–36 km'
        caption = 'The highlighted surface traces where the model reaches 10 Ω·m.'
    if t >= 76:
        q = smooth((t-76)/7)
        az = 60-120*q
        planes = [('e', 0., q), ('n', 0., q)]
        iso_alpha = 1.
        title, detail = 'Surface context and crustal structure', 'Perpendicular sections with the 10 Ω·m isosurface'
        caption = 'Surface context, cross-sections and 3D geometry bring the resistivity model together.'
    if t >= 85:
        title, detail = 'Kanchanaburi 3D MT model', 'The first MT crustal structure in Thailand'
        caption = 'Kanchanaburi 3D MT model: The first MT crustal structure in Thailand'
    return az, el, planes, iso_alpha, overlay_alpha, box_alpha, title, detail, caption, depth, section


def draw_model(im, state):
    az, el, planes, iso_alpha, ov_alpha, box_alpha = state[:6]
    basis = camera(az, el)
    draw = ImageDraw.Draw(im)
    polygons, colors, distances = [], [], []
    for kind, value, alpha in planes:
        if alpha < .001:
            continue
        verts, cols = {'z': plane_z, 'e': plane_e, 'n': plane_n}[kind](value)
        xy, dep = project(verts, basis, el)
        polygons.extend(xy)
        colors.extend(np.round(cols*alpha+np.array(BG)*(1-alpha)).astype(np.uint8))
        distances.extend(dep.mean(axis=1))
    if iso_alpha > .001:
        xy, dep = project(ISO_V, basis, el)
        light = np.array([.3, -.5, .81]); light /= np.linalg.norm(light)
        shade = .62+.38*np.abs(ISO_NORMAL @ light)
        base = rgb(1.).astype(float)
        cols = (base[None, :]*shade[:, None])*iso_alpha+np.array(BG)*(1-iso_alpha)
        polygons.extend(xy)
        colors.extend(np.round(cols).astype(np.uint8))
        distances.extend(dep.mean(axis=1))
    if len(polygons):
        order = np.argsort(distances, kind='stable')
        for i in order:
            draw.polygon([tuple(p) for p in polygons[i]], fill=tuple(colors[i]))
    if box_alpha > .001:
        c = mix_color((161, 173, 182), box_alpha*.65)
        for zz in [0, -36]:
            pts = np.array([[-80,-80,zz],[80,-80,zz],[80,80,zz],[-80,80,zz],[-80,-80,zz]])
            xy, _ = project(pts, basis, el); draw.line([tuple(p) for p in xy], fill=c, width=2)
        for east in [-80, 80]:
            for north in [-80, 80]:
                xy, _ = project(np.array([[east,north,0],[east,north,-36]]), basis, el)
                draw.line([tuple(p) for p in xy], fill=c, width=2)
        # Positive depth labels despite internal negative-z rendering coordinates.
        for depth in [0, 9, 18, 27, 36]:
            xy, _ = project(np.array([[80,-80,-depth]]), basis, el)
            x,y = xy[0]; draw.text((x+8,y-10), str(depth), font=FONTS[20], fill=MUTED)
        xy, _ = project(np.array([[80,-80,-18]]), basis, el)
        draw.text((xy[0,0]+45,xy[0,1]-10), 'Depth (km)', font=FONTS[20], fill=MUTED)
    if ov_alpha > .001:
        for pts, color, width, _ in overlays:
            xy, _ = project(pts, basis, el)
            draw.line([tuple(p) for p in xy], fill=mix_color(color, ov_alpha), width=width)
    xy, _ = project(stations, basis, el)
    for x,y in xy:
        draw.polygon([(x-5,y-4),(x+5,y-4),(x,y+6)], fill=INK)
    # Orientation triad and a local-distance scale, anchored outside the volume.
    origin = np.array([175., 802.])
    for vector, label in [(np.array([15,0,0]),'E'),(np.array([0,15,0]),'N')]:
        direction = vector @ basis[:2].T * np.array([3.,-3.])
        tip = origin+direction
        draw.line([tuple(origin), tuple(tip)], fill=INK, width=3)
        draw.ellipse((tip[0]-3,tip[1]-3,tip[0]+3,tip[1]+3),fill=INK)
        draw.text(tuple(tip+[7,-13]), label, font=BOLD[24], fill=INK)
    draw.text((95,855), 'True spatial proportions · all distances in km', font=FONTS[20], fill=MUTED)


MAP_BOX = (1440, 260, 1832, 652)


def map_xy(pts):
    return np.stack([MAP_BOX[0]+(pts[...,0]+80)/160*(MAP_BOX[2]-MAP_BOX[0]),
                     MAP_BOX[3]-(pts[...,1]+80)/160*(MAP_BOX[3]-MAP_BOX[1])], -1)


def make_map():
    im = Image.new('RGB',(W,H),BG); draw=ImageDraw.Draw(im)
    for v,c in zip(surface_v,surface_c):
        draw.polygon([tuple(p) for p in map_xy(v)],fill=tuple(c))
    for pts,c,_,_ in overlays:
        draw.line([tuple(p) for p in map_xy(pts)],fill=c,width=2)
    for x,y in map_xy(stations):
        draw.polygon([(x-3,y-3),(x+3,y-3),(x,y+4)],fill=INK)
    draw.rectangle(MAP_BOX,outline=(167,181,191),width=2)
    return im.crop(MAP_BOX)


MAP_IMAGE = make_map()


def frame(t):
    state=section_state(t)
    im=Image.new('RGB',(W,H),BG)
    draw=ImageDraw.Draw(im)
    draw.rectangle((60,53,67,125),fill=ACCENT)
    draw.text((88,46),state[6],font=BOLD[42],fill=INK)
    draw.text((90,103),state[7],font=FONTS[28],fill=MUTED)
    draw.line((60,159,1860,159),fill=(214,222,229),width=2)
    draw_model(im,state)
    draw=ImageDraw.Draw(im)
    draw.rounded_rectangle((1390,188,1870,885),radius=20,fill=(239,243,246),outline=(220,228,234),width=2)
    draw.text((1440,214),'SURVEY CONTEXT',font=BOLD[24],fill=INK)
    im.paste(MAP_IMAGE,MAP_BOX[:2])
    sec=state[-1]
    if sec:
        direction,value=sec
        pts=np.array([[value,-80],[value,80]]) if direction=='e' else np.array([[-80,value],[80,value]])
        draw.line([tuple(p) for p in map_xy(pts)],fill=(255,255,255),width=7)
        draw.line([tuple(p) for p in map_xy(pts)],fill=INK,width=3)
    draw.text((1845,262),'N',font=BOLD[20],fill=INK)
    draw.text((1438,664),'−80',font=FONTS[18],fill=MUTED)
    draw.text((1613,664),'0',font=FONTS[18],fill=MUTED)
    draw.text((1798,664),'+80 km E',font=FONTS[18],fill=MUTED)
    draw.text((1440,690),'▼ Stations',font=FONTS[18],fill=INK)
    draw.text((1552,690),'— Faults',font=FONTS[18],fill=(179,49,100))
    draw.text((1660,690),'— Reservoirs',font=FONTS[18],fill=(22,116,180))
    draw.text((1440,711),'Resistivity (Ω·m) · logarithmic',font=BOLD[24],fill=INK)
    bar=np.tile(rgb(np.linspace(-1,4,392))[None,:,:],(20,1,1))
    im.paste(Image.fromarray(bar),(1440,753))
    for i,label in enumerate(['0.1','1','10','100','1,000','10,000']):
        x=1440+i*392/5
        draw.text((x,781),label,font=FONTS[18],fill=INK,anchor='mt')
    draw.text((1440,819),'Conductive',font=FONTS[20],fill=MUTED)
    draw.text((1832,819),'Resistive',font=FONTS[20],fill=MUTED,anchor='rt')
    draw.rectangle((60,912,1860,1008),fill=(234,240,244))
    closing=t>=85
    f=BOLD[30] if closing else FONTS[28]
    draw.text((960,954),state[8],font=f,fill=INK,anchor='mm')
    draw.text((60,1020),'3D MT resistivity model · Kanchanaburi, Thailand',font=FONTS[20],fill=MUTED)
    draw.text((1860,1020),f'{int(t):02d} / 90 s',font=FONTS[20],fill=MUTED,anchor='rt')
    draw.rectangle((60,1058,1860,1063),fill=(218,228,234))
    draw.rectangle((60,1058,60+1800*t/90,1063),fill=ACCENT)
    # Fade in/out while retaining a readable four-second closing hold.
    fade=min(smooth(t/.8),smooth((90-t)/1.))
    if fade<1:
        im=Image.blend(Image.new('RGB',(W,H),BG),im,float(fade))
    return im


def metadata():
    return dict(model=str(model.source.relative_to(ROOT)),dimensions=list(model.shape),
                full_extent_km=[348,348,129],view_extent_km=[160,160,36],
                resistivity_display_range_ohm_m=[.1,10000],colormap='RdYlBu',
                isosurface_ohm_m=10,isosurface_depth_interval_km=[3.5,36],
                isosurface_interpolation='VTK cell-to-point averaging in log10 resistivity',
                station_source='modPAPER/kansea4_resp.01_01',station_count=39,
                maximum_station_difference_from_matlab_m=station_error,
                closing_caption='Kanchanaburi 3D MT model: The first MT crustal structure in Thailand',
                width=W,height=H,fps=FPS,duration_s=DURATION,audio=False,
                rendering='CPU orthographic projection; original cell values on slices',
                model_header=model.header)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT/'render_settings.json').write_text(json.dumps(metadata(),indent=2)+'\n')
    samples=[4,14,24,38,50,58,70,81,87]
    for t in samples:
        frame(t).save(OUT/f'preview_{t:02d}s.jpg',quality=92)
    if args.preview:
        return
    dest=OUT/'Kanchanaburi_3D_MT_model.mp4'
    temp=OUT/'Kanchanaburi_3D_MT_model.rendering.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
         '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
         '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    start=time.monotonic()
    try:
        for k in range(FPS*DURATION):
            proc.stdin.write(frame(k/FPS).tobytes())
            if k%150==0:
                print(f'Frame {k}/{FPS*DURATION} · {k/FPS:.0f}s of 90s · elapsed {time.monotonic()-start:.1f}s',flush=True)
        proc.stdin.close()
        if proc.wait()!=0:
            raise RuntimeError('ffmpeg encoding failed')
        temp.replace(dest)
    except BaseException:
        proc.kill();proc.wait();raise
    print(f'Completed: {dest}',flush=True)


if __name__=='__main__':
    main()
