use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{TensorData, activation::tanh},
};

type MyBackend = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
struct Net<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    output: Linear<B>,
}

impl<B: Backend> Net<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            // 3 inputs -> 4 hidden neurons
            layer1: LinearConfig::new(3, 4).init(device),

            // 4 hidden neurons -> 4 hidden neurons
            layer2: LinearConfig::new(4, 4).init(device),

            // 4 hidden neurons -> 1 output
            output: LinearConfig::new(4, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(x);
        let x = tanh(x);

        let x = self.layer2.forward(x);
        let x = tanh(x);

        self.output.forward(x)
    }
}

#[test]
fn micrograd_burn() {
    let device = Default::default();

    let mut model = Net::<MyBackend>::new(&device);
    let mut optimizer = AdamConfig::new().init();

    let learning_rate = 0.01;

    // Shape: [4 samples, 3 inputs]
    let xs = Tensor::<MyBackend, 2>::from_data(
        TensorData::from([
            [2.0_f32, 3.0, -1.0],
            [3.0_f32, -1.0, 0.5],
            [0.5_f32, 1.0, 1.0],
            [1.0_f32, 1.0, -1.0],
        ]),
        &device,
    );

    // Shape: [4 samples, 1 output]
    let ys = Tensor::<MyBackend, 2>::from_data(
        TensorData::from([[1.0_f32], [-1.0_f32], [1.0_f32], [1.0_f32]]),
        &device,
    );

    for epoch in 0..10_000 {
        let estimated_y = model.forward(xs.clone());

        // mean((estimated_y - y)^2)
        let diff = estimated_y - ys.clone();
        let loss = diff.clone().powf_scalar(2.0).mean();

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, grads);

        if epoch % 1000 == 0 {
            let loss_value = loss.into_data().to_vec::<f32>().unwrap()[0];

            println!("epoch {epoch}, loss = {loss_value}");
        }
    }

    let predictions = model.forward(xs);

    println!("predictions:");
    println!("{:?}", predictions.into_data());
}
