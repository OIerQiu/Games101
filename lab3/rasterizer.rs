use std::rc::Rc;

use nalgebra::{Matrix4, Vector2, Vector3, Vector4};
use crate::shader::{FragmentShaderPayload, VertexShaderPayload};
use crate::texture::Texture;
use crate::triangle::Triangle;

#[allow(dead_code)]
pub enum Buffer {
    Color,
    Depth,
    Both,
}

#[allow(dead_code)]
pub enum Primitive {
    Line,
    Triangle,
}

#[derive(Default)]
pub struct Rasterizer {
    model: Matrix4<f64>,
    view: Matrix4<f64>,
    projection: Matrix4<f64>,
    texture: Option<Texture>,

    vert_shader: Option<fn(&VertexShaderPayload) -> Vector3<f64>>,
    fragment_shader: Option<fn(&FragmentShaderPayload) -> Vector3<f64>>,
    frame_buf: Vec<Vector3<f64>>,
    depth_buf: Vec<f64>,
    width: u64,
    height: u64,
}

#[derive(Clone, Copy)]
pub struct PosBufId(usize);

#[derive(Clone, Copy)]
pub struct IndBufId(usize);

#[derive(Clone, Copy)]
pub struct ColBufId(usize);

impl Rasterizer {
    pub fn new(w: u64, h: u64) -> Self {
        let mut r = Rasterizer::default();
        r.width = w;
        r.height = h;
        r.frame_buf.resize((w * h) as usize, Vector3::zeros());
        r.depth_buf.resize((w * h) as usize, 0.0);
        r.texture = None;
        r
    }

    fn get_index(height: u64, width: u64, x: usize, y: usize) -> usize {
        ((height - 1 - y as u64) * width + x as u64) as usize
    }

    fn set_pixel(height: u64, width: u64, frame_buf: &mut Vec<Vector3<f64>>, point: &Vector3<f64>, color: &Vector3<f64>) {
        let ind = (height as f64 - 1.0 - point.y) * width as f64 + point.x;
        frame_buf[ind as usize] = *color;
    }

    pub fn clear(&mut self, buff: Buffer) {
        match buff {
            Buffer::Color =>
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0)),
            Buffer::Depth =>
                self.depth_buf.fill(f64::MAX),
            Buffer::Both => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_buf.fill(f64::MAX);
            }
        }
    }
    pub fn set_model(&mut self, model: Matrix4<f64>) {
        self.model = model;
    }

    pub fn set_view(&mut self, view: Matrix4<f64>) {
        self.view = view;
    }

    pub fn set_projection(&mut self, projection: Matrix4<f64>) {
        self.projection = projection;
    }

    pub fn set_texture(&mut self, tex: Texture) { 
        self.texture = Some(tex); 
    }

    pub fn set_vertex_shader(&mut self, vert_shader: fn(&VertexShaderPayload) -> Vector3<f64>) {
        self.vert_shader = Some(vert_shader);
    }
    
    pub fn set_fragment_shader(&mut self, frag_shader: fn(&FragmentShaderPayload) -> Vector3<f64>) {
        self.fragment_shader = Some(frag_shader);
    }

    pub fn draw(&mut self, triangles: &Vec<Triangle>) {
        let mvp = self.projection * self.view * self.model;

        // 遍历每个小三角形
        for triangle in triangles { 
            self.rasterize_triangle(&triangle, mvp); 
        }
    }

    pub fn rasterize_triangle(&mut self, triangle: &Triangle, mvp: Matrix4<f64>) {
        /*  Implement your code here  */
        let (t, v) = Rasterizer::get_new_tri(triangle, self.view, self.model, mvp, (self.width, self.height));
        let xmin = std::cmp::min(t.v[0].x as usize,std::cmp::min(t.v[1].x as usize,t.v[2].x as usize));
        let xmax = std::cmp::max(t.v[0].x as usize,std::cmp::max(t.v[1].x as usize,t.v[2].x as usize));
        let ymin = std::cmp::min(t.v[0].y as usize,std::cmp::min(t.v[1].y as usize,t.v[2].y as usize));
        let ymax = std::cmp::max(t.v[0].y as usize,std::cmp::max(t.v[1].y as usize,t.v[2].y as usize));
        for x in xmin..xmax+1 {
            for y in ymin..ymax+1 {
                if inside_triangle(x as f64, y as f64, &t.v) {
                    let nx = (t.v[1].y - t.v[0].y)*(t.v[2].z - t.v[0].z) - (t.v[1].z - t.v[0].z)*(t.v[2].y - t.v[0].y);
	                let ny = (t.v[1].z - t.v[0].z)*(t.v[2].x - t.v[0].x) - (t.v[1].x - t.v[0].x)*(t.v[2].z - t.v[0].z);
	                let nz = (t.v[1].x - t.v[0].x)*(t.v[2].y - t.v[0].y) - (t.v[1].y - t.v[0].y)*(t.v[2].x - t.v[0].x);
                    let mut z = 0.0;
                    if (nz != 0.0)		
                    {
                        z = t.v[0].z - (nx * (x as f64 - t.v[0].x) + ny * (y as f64 - t.v[0].y)) / nz;
                    }
                    let index = Rasterizer::get_index(self.height, self.width, x, y);
                    if z < self.depth_buf[index] {
                        let mut a = [0.0;3];
                        for i in 0..3 {
                            let x0 = t.v[i].x;
                            let mut y0 = t.v[i].y;
                            let x1 = t.v[(i+1)%3].x;
                            let y1 = t.v[(i+1)%3].y;
                            let x2 = t.v[(i+2)%3].x;
                            let y2 = t.v[(i+2)%3].y;
                            if x1 == x2 {
                                a[i] = (x as f64-x1)/(x0 - x1);
                            }
                            else {
                                let y12 = ( y1 * (x as f64- x2) + y2 * (x1 - x as f64) ) / (x1 - x2);
                                y0 = y0 + ((y1 - y2)/(x1 - x2))*(x as f64-x0);
                                a[i] = (y as f64 - y12)/(y0 - y12);
                            }
                        }       
                        let t_color = t.color[0]*a[0]+t.color[1]*a[1]+t.color[2]*a[2];
                        let t_normal = t.normal[0]*a[0]+t.normal[1]*a[1]+t.normal[2]*a[2];
                        let t_tex_coords = t.tex_coords[0]*a[0]+t.tex_coords[1]*a[1]+t.tex_coords[2]*a[2];
                        let payload = FragmentShaderPayload::new(&t_color, &t_normal, &t_tex_coords, match &self.texture { None =>None, Some(tex)=>Some(Rc::new(tex))});
                        let mut color = Vector3::zeros();
                        if let Some(shader) = self.fragment_shader {
                            color = shader(&payload);
                        }
                        self.depth_buf[index] = z;
                        self.frame_buf[index].x = color.x;
                        self.frame_buf[index].y = color.y;
                        self.frame_buf[index].z = color.z;
                    }
                }
            }
        }
    }
    
    fn interpolate_Vec3(a: f64, b: f64, c: f64, vert1: Vector3<f64>, vert2: Vector3<f64>, vert3: Vector3<f64>, weight: f64) -> Vector3<f64> {
        (a * vert1 + b * vert2 + c * vert3) / weight
    }
    fn interpolate_Vec2(a: f64, b: f64, c: f64, vert1: Vector2<f64>, vert2: Vector2<f64>, vert3: Vector2<f64>, weight: f64) -> Vector2<f64> {
        (a * vert1 + b * vert2 + c * vert3) / weight
    }

    fn get_new_tri(t: &Triangle, view: Matrix4<f64>, model: Matrix4<f64>, mvp: Matrix4<f64>,
                    (width, height): (u64, u64)) -> (Triangle, Vec<Vector3<f64>>) {
        let f1 = (50.0 - 0.1) / 2.0; // zfar和znear距离的一半
        let f2 = (50.0 + 0.1) / 2.0; // zfar和znear的中心z坐标
        let mut new_tri = (*t).clone();
        let mm: Vec<Vector4<f64>> = (0..3).map(|i| view * model * t.v[i]).collect();
        let view_space_pos: Vec<Vector3<f64>> = mm.iter().map(|v| v.xyz()).collect();
        let mut v: Vec<Vector4<f64>> = (0..3).map(|i| mvp * t.v[i]).collect();

        // 换算齐次坐标
        for vec in v.iter_mut() {
            vec.x /= vec.w;
            vec.y /= vec.w;
            vec.z /= vec.w;
        }
        let inv_trans = (view * model).try_inverse().unwrap().transpose();
        let n: Vec<Vector4<f64>> = (0..3).map(|i| inv_trans * to_vec4(t.normal[i], Some(0.0))).collect();

        // 视口变换得到顶点在屏幕上的坐标, 即screen space
        for vert in v.iter_mut() {
            vert.x = 0.5 * width as f64 * (vert.x + 1.0);
            vert.y = 0.5 * height as f64 * (vert.y + 1.0);
            vert.z = vert.z * f1 + f2;
        }
        for i in 0..3 {
            new_tri.set_vertex(i, v[i]);
        }
        for i in 0..3 {
            new_tri.set_normal(i, n[i].xyz());
        }

        new_tri.set_color(0, 148.0, 121.0, 92.0);
        new_tri.set_color(1, 148.0, 121.0, 92.0);
        new_tri.set_color(2, 148.0, 121.0, 92.0);

        (new_tri, view_space_pos)
    }

    pub fn frame_buffer(&self) -> &Vec<Vector3<f64>> {
        &self.frame_buf
    }

}

fn to_vec4(v3: Vector3<f64>, w: Option<f64>) -> Vector4<f64> {
    Vector4::new(v3.x, v3.y, v3.z, w.unwrap_or(1.0))
}

fn inside_triangle(x: f64, y: f64, v: &[Vector4<f64>; 3]) -> bool {
    let v = [
        Vector3::new(v[0].x, v[0].y, 1.0),
        Vector3::new(v[1].x, v[1].y, 1.0),
        Vector3::new(v[2].x, v[2].y, 1.0), ];

    let f0 = v[1].cross(&v[0]);
    let f1 = v[2].cross(&v[1]);
    let f2 = v[0].cross(&v[2]);
    let p = Vector3::new(x, y, 1.0);
    if (p.dot(&f0) * f0.dot(&v[2]) > 0.0) &&
        (p.dot(&f1) * f1.dot(&v[0]) > 0.0) &&
        (p.dot(&f2) * f2.dot(&v[1]) > 0.0) {
        true
    } else {
        false
    }
}

fn compute_barycentric2d(x: f64, y: f64, v: &[Vector4<f64>; 3]) -> (f64, f64, f64) {
    let c1 = (x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * y + v[1].x * v[2].y - v[2].x * v[1].y) / (v[0].x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * v[0].y + v[1].x * v[2].y - v[2].x * v[1].y);
    let c2 = (x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * y + v[2].x * v[0].y - v[0].x * v[2].y) / (v[1].x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * v[1].y + v[2].x * v[0].y - v[0].x * v[2].y);
    let c3 = (x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * y + v[0].x * v[1].y - v[1].x * v[0].y) / (v[2].x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * v[2].y + v[0].x * v[1].y - v[1].x * v[0].y);
    (c1, c2, c3)
}