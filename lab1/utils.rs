use std::os::raw::c_void;
use nalgebra::{Matrix4, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};

pub type V3d = Vector3<f64>;

pub(crate) fn get_view_matrix(eye_pos: V3d) -> Matrix4<f64> {
    let mut view: Matrix4<f64> = Matrix4::identity();
    /*  implement your code here  */    
    view = Matrix4::new(
        1.0, 0.0, 0.0, -eye_pos.x,
        0.0, 1.0, 0.0, -eye_pos.y,
        0.0, 0.0, 1.0, -eye_pos.z,
        0.0, 0.0, 0.0, 1.0,
    );

    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> Matrix4<f64> {
    let mut model: Matrix4<f64> = Matrix4::identity();
    /*  implement your code here  */
    let angle:f64 = rotation_angle*std::f64::consts::PI / 180.0;
    model = Matrix4::new(
        angle.cos(), -angle.sin(), 0.0, 0.0,
        angle.sin(), angle.cos(), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    model
}

pub(crate) fn get_projection_matrix(eye_fov: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Matrix4<f64> {
    let mut projection: Matrix4<f64> = Matrix4::identity();
    /*  implement your code here  */
    let halve_angle:f64 = -(eye_fov/2.0)*std::f64::consts::PI / 180.0;
    let top:f64 = halve_angle.tan() * z_near.abs();
    let bottom:f64 = -top;
    let right:f64 = top * aspect_ratio;
    let left:f64 = -right;
    let o:Matrix4<f64> = Matrix4::new(
        2.0/(right-left), 0.0, 0.0, 0.0,
        0.0, 2.0/(top-bottom), 0.0, 0.0,
        0.0, 0.0, 2.0/(z_near-z_far), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let p:Matrix4<f64> = Matrix4::new(
        1.0, 0.0, 0.0, -(right+left)/2.0,
        0.0, 1.0, 0.0, -(top+bottom)/2.0,
        0.0, 0.0, 1.0, -(z_near+z_far)/2.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let q:Matrix4<f64> = Matrix4::new(
        z_near, 0.0, 0.0, 0.0,
        0.0, z_near, 0.0, 0.0,
        0.0, 0.0, z_near+z_far, -z_near * z_far,
        0.0, 0.0, 1.0, 0.0,
    );
    projection = o * p * q;

    projection
}

pub(crate) fn get_rotation(axis: Vector3<f64>, angle: f64) -> Matrix4<f64> {
    let mut arbitrary_rotation: Matrix4<f64> = Matrix4::identity();
    /*  implement your code here  */
    let arch:f64 = angle*std::f64::consts::PI / 180.0;
    let r: Matrix4<f64> = Matrix4::new(
        axis.x*axis.x, axis.x*axis.y, axis.x*axis.z, 0.0,
        axis.x*axis.y, axis.y*axis.y, axis.y*axis.z, 0.0,
        axis.x*axis.z, axis.y*axis.z, axis.z*axis.z, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let s: Matrix4<f64> = Matrix4::new(
        0.0, -axis.z, axis.y, 0.0,
        axis.z, 0.0, -axis.x, 0.0,
        -axis.y, axis.x, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    );
    arbitrary_rotation = arch.cos()*Matrix4::identity() + (1.0-arch.cos())*r + arch.sin()*s;

    arbitrary_rotation
}

pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<V3d>) -> opencv::core::Mat {
    let mut image = unsafe {
        Mat::new_rows_cols_with_data(
            700, 700,
            opencv::core::CV_64FC3,
            frame_buffer.as_ptr() as *mut c_void,
            opencv::core::Mat_AUTO_STEP,
        ).unwrap()
    };
    let mut img = Mat::copy(&image).unwrap();
    image.convert_to(&mut img, opencv::core::CV_8UC3, 1.0, 1.0).expect("panic message");
    cvt_color(&img, &mut image, COLOR_RGB2BGR, 0).unwrap();
    image
}
