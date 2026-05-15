//! Quick implementation of
//! Andrej Karpathy's The spelled-out intro to neural networks and backpropagation: building micrograd
//! https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2

#![allow(unused)]

use std::ops::{Index, IndexMut};

use rand::RngExt;

type ExprId = usize;

#[derive(Debug, Clone, Copy)]
enum Op {
    Constant(f64),
    Sum(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Tanh(ExprId),
    Exp(ExprId),
    Pow(ExprId, ExprId),
}
impl Op {
    fn sources(&self) -> [Option<ExprId>; 2] {
        match self {
            Op::Constant(_) => [None, None],
            Op::Sum(l, r) => [Some(*l), Some(*r)],
            Op::Mul(l, r) => [Some(*l), Some(*r)],
            Op::Tanh(x) => [Some(*x), None],
            Op::Exp(x) => [Some(*x), None],
            Op::Pow(x, n) => [Some(*x), Some(*n)],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Expr {
    data: f64,
    grad: f64,
    op: Op,
}

#[derive(Default)]
struct ExprArena {
    exprs: Vec<Expr>,
}

impl ExprArena {
    fn push(&mut self, op: Op, data: f64) -> ExprId {
        let id = self.exprs.len();
        let expr = Expr {
            grad: 0.0,
            data,
            op,
        };
        self.exprs.push(expr);
        id
    }
    pub fn v<TI: Into<f64>>(&mut self, t: TI) -> ExprId {
        let value = t.into();
        self.push(Op::Constant(value), value)
    }
    pub fn va<const N: usize, TI: Into<f64> + Copy>(&mut self, t: [TI; N]) -> [ExprId; N] {
        t.map(|t| self.push(Op::Constant(t.into()), t.into()))
    }
    pub fn m(&mut self, lhe: ExprId, rhe: ExprId) -> ExprId {
        let value = self.exprs[lhe].data * self.exprs[rhe].data;
        self.push(Op::Mul(lhe, rhe), value)
    }
    pub fn s(&mut self, lhe: ExprId, rhe: ExprId) -> ExprId {
        let value = self.exprs[lhe].data + self.exprs[rhe].data;
        self.push(Op::Sum(lhe, rhe), value)
    }
    pub fn tanh(&mut self, x: ExprId) -> ExprId {
        let e2x = f64::exp(2.0 * self.exprs[x].data);
        let value = (e2x - 1.0) / (e2x + 1.0);
        self.push(Op::Tanh(x), value)
    }
    pub fn exp(&mut self, x: ExprId) -> ExprId {
        let value = f64::exp(self.exprs[x].data);
        self.push(Op::Exp(x), value)
    }
    pub fn pow(&mut self, x: ExprId, n: ExprId) -> ExprId {
        let value = self.exprs[x].data.powf(self.exprs[n].data);
        self.push(Op::Pow(x, n), value)
    }
    pub fn neg(&mut self, x: ExprId) -> ExprId {
        let m1 = self.v(-1.0);
        self.m(x, m1)
    }
    pub fn sub(&mut self, x: ExprId, y: ExprId) -> ExprId {
        let neg_y = self.neg(y);
        self.s(x, neg_y)
    }
    pub fn div(&mut self, x: ExprId, y: ExprId) -> ExprId {
        let m1 = self.v(-1.0);
        let inv_y = self.pow(y, m1);
        self.m(x, inv_y)
    }

    fn topo_sorted_nodes(&self, start: ExprId) -> Vec<ExprId> {
        let mut seen = vec![false; self.exprs.len()];
        let mut stack = vec![start];
        let mut list = Vec::with_capacity(self.exprs.len());

        while !stack.is_empty() {
            let expr_id = stack.pop().unwrap();
            if seen[expr_id] {
                continue;
            };
            seen[expr_id] = true;

            for src in self.exprs[expr_id].op.sources() {
                if let Some(src) = src {
                    stack.push(src);
                }
            }
            list.push(expr_id);
        }
        list
    }

    // Backward compute partial derivates
    pub fn backward(&mut self, end: ExprId) -> () {
        self.exprs[end].grad = 1.0;

        for x in self.topo_sorted_nodes(end) {
            let x = self.exprs[x];
            match x.op {
                Op::Constant(_) => {}
                Op::Mul(lhe, rhe) => {
                    self.exprs[lhe].grad = self.exprs[lhe].grad + self.exprs[rhe].data * x.grad;
                    self.exprs[rhe].grad = self.exprs[rhe].grad + self.exprs[lhe].data * x.grad;
                }
                Op::Sum(lhe, rhe) => {
                    self.exprs[lhe].grad += x.grad;
                    self.exprs[rhe].grad += x.grad;
                }
                Op::Tanh(e) => {
                    // 1 - tanh(x)^2
                    self.exprs[e].grad += (1.0 - (x.data * x.data)) * x.grad;
                }
                Op::Exp(e) => {
                    self.exprs[e].grad += x.data * x.grad;
                }
                Op::Pow(e, n) => {
                    let deriv =
                        self.exprs[n].data * self.exprs[e].data.powf(self.exprs[n].data - 1.0);
                    self.exprs[e].grad += deriv * x.grad;
                }
            }
        }
    }
}

impl Index<usize> for ExprArena {
    type Output = Expr;

    fn index(&self, index: usize) -> &Self::Output {
        &self.exprs[index]
    }
}
impl IndexMut<usize> for ExprArena {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.exprs[index]
    }
}

struct Neuron {
    weights: Vec<ExprId>,
    bias: ExprId,
}

impl Neuron {
    fn new(arena: &mut ExprArena, input_count: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..input_count)
            .map(|_| arena.v(rng.random_range(-1.0..1.0)))
            .collect();
        let bias = arena.v(rng.random_range(-1.0..1.0));
        Self { weights, bias }
    }
    fn forward(&self, arena: &mut ExprArena, x: &[ExprId]) -> ExprId {
        assert_eq!(self.weights.len(), x.len());
        let mut sum = self.bias;
        for (w, x) in self.weights.iter().zip(x.iter()) {
            let m = arena.m(*w, *x);
            sum = arena.s(sum, m);
        }
        arena.tanh(sum)
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(arena: &mut ExprArena, input_count: usize, output_count: usize) -> Self {
        let neurons = (0..output_count)
            .map(|_| Neuron::new(arena, input_count))
            .collect();
        Self { neurons }
    }
    fn forward(&self, arena: &mut ExprArena, x: &[ExprId]) -> Vec<ExprId> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(arena, x))
            .collect()
    }
}

struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}
impl MultiLayerPerceptron {
    fn new(arena: &mut ExprArena, input_count: usize, output_counts: &[usize]) -> Self {
        assert_eq!(
            1,
            *output_counts.last().unwrap(),
            "This kind of perception needs last layer to have only 1 neuron"
        );
        let output_counts_len = output_counts.len();
        let neurons_per_layer: Vec<_> =
            std::iter::once(&input_count).chain(output_counts).collect();
        let layers = (0..output_counts_len)
            .map(|i| Layer::new(arena, *neurons_per_layer[i], *neurons_per_layer[i + 1]))
            .collect();
        Self { layers }
    }
    fn forward(&self, arena: &mut ExprArena, x: &[ExprId]) -> Vec<ExprId> {
        let mut x = x.to_vec();
        for layer in &self.layers {
            x = layer.forward(arena, &x)
        }
        x
    }
    fn parameters(&self) -> Vec<ExprId> {
        let mut params = Vec::new();
        for layer in &self.layers {
            for neuron in &layer.neurons {
                params.append(&mut neuron.weights.clone());
                params.push(neuron.bias);
            }
        }
        params
    }
    fn train(
        &self,
        arena: &mut ExprArena,
        training: Vec<(Vec<f64>, f64)>,
        gradient_increment: f64,
        loss_limit: f64,
        count_limit: usize,
    ) -> f64 {
        let arena_training_nodes_start = arena.exprs.len();

        let training_x: Vec<Vec<usize>> = training
            .iter()
            .map(|(x, _y)| x.iter().map(|xi| arena.v(*xi)).collect())
            .collect();
        let training_y: Vec<usize> = training.iter().map(|(_x, y)| arena.v(*y)).collect();

        let parameters = self.parameters();

        let arena_training_nodes_end = arena.exprs.len();

        let mut count = 1;
        let mut current_loss = loss_limit + 1.0;

        while count < count_limit && current_loss > loss_limit {
            arena.exprs.truncate(arena_training_nodes_end);

            let mut loss = arena.v(0);
            let two = arena.v(2);
            for (x, y) in training_x.iter().zip(training_y.iter()) {
                let y_estimate = self.forward(arena, x)[0];
                let y_diff = arena.sub(*y, y_estimate);
                let y_diff_sq = arena.pow(y_diff, two);
                loss = arena.s(loss, y_diff_sq);
            }

            for expr in arena.exprs.iter_mut() {
                expr.grad = 0.0;
            }
            arena.backward(loss);

            for param in &parameters {
                arena[*param].data += -gradient_increment * arena[*param].grad;
            }

            current_loss = arena[loss].data;

            count += 1;
        }

        arena.exprs.truncate(arena_training_nodes_start);

        current_loss
    }
}

#[test]
fn test_micrograd_base() {
    // https://youtu.be/VMj-3S1tku0?si=yQkbFeaZga8gK-4G&t=4945

    let trunc4 = |f: f64| (f * 10_000.0).trunc() / 10000.0;

    let mut ea = ExprArena::default();

    let x1 = ea.v(2.0);
    let w1 = ea.v(-3);
    let x2 = ea.v(0.0);
    let w2 = ea.v(1);
    let x1w1 = ea.m(x1, w1);
    let x2w2 = ea.m(x2, w2);
    let x1w1x2w2 = ea.s(x1w1, x2w2);
    let b = ea.v(6.88137358);
    let n = ea.s(x1w1x2w2, b);
    let o = ea.tanh(n);
    ea.backward(o);

    assert_eq!(0.7071, trunc4(ea[o].data));
    assert_eq!(0.5, trunc4(ea[n].grad));
    assert_eq!(-1.5, trunc4(ea[x1].grad));
    assert_eq!(0.5, trunc4(ea[x2].grad));
    assert_eq!(1.0, trunc4(ea[w1].grad));
    assert_eq!(0.0, trunc4(ea[w2].grad));
}

#[test]
fn test_div_sub() {
    let mut ea = ExprArena::default();

    let x = ea.v(6.0);
    let y = ea.v(2.0);
    let div = ea.div(x, y);
    let sub = ea.sub(x, y);
    assert_eq!(3.0, ea[div].data);
    assert_eq!(4.0, ea[sub].data);
}

#[test]
fn test_mlp_base() {
    let mut ea = ExprArena::default();
    let mlp = MultiLayerPerceptron::new(&mut ea, 3, &[4, 4, 1]);
    let xs = [
        ea.va([2.0, 3.0, -1.0]),
        ea.va([3.0, -1.0, 0.5]),
        ea.va([0.5, 1.0, 1.0]),
        ea.va([1.0, 1.0, -1.0]),
    ];
    // desired targets
    let ys = ea.va([1.0, -1.0, 1.0, 1.0]);

    let parameters = mlp.parameters();
    // https://youtu.be/VMj-3S1tku0?si=c7g8c9PMsgFj7m78&t=7268
    assert_eq!(parameters.len(), 41);

    let ea_expr_limit = ea.exprs.len();

    let mut count = 1;
    loop {
        ea.exprs.truncate(ea_expr_limit);

        let mut loss = ea.v(0);
        let two = ea.v(2);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let y_estimate = mlp.forward(&mut ea, x)[0];
            let y_diff = ea.sub(*y, y_estimate);
            let y_diff_sq = ea.pow(y_diff, two);
            loss = ea.s(loss, y_diff_sq);
        }

        for expr in ea.exprs.iter_mut() {
            expr.grad = 0.0;
        }
        ea.backward(loss);

        for param in &parameters {
            ea[*param].data += -0.05 * ea[*param].grad;
        }

        count += 1;
        if ea[loss].data < 0.0001 {
            println!("loss={} count={}", ea[loss].data, count);
            break;
        }
    }
}

#[test]
fn test_mlp_training() {
    let mut ea = ExprArena::default();
    let mlp = MultiLayerPerceptron::new(&mut ea, 3, &[4, 4, 1]);
    let training_data = vec![
        (vec![2.0, 3.0, -1.0], 1.0),
        (vec![3.0, -1.0, 0.5], -1.0),
        (vec![0.5, 1.0, 1.0], 1.0),
        (vec![1.0, 1.0, -1.0], 1.0),
    ];
    let gradient_increment = 0.05;
    let loss_limit = 0.0001;
    let count_limit = 1_000_000;
    let loss = mlp.train(
        &mut ea,
        training_data,
        gradient_increment,
        loss_limit,
        count_limit,
    );
    assert!(loss < loss_limit);

    let data = ea.va([2.0, 3.0, -1.0]);
    let res = mlp.forward(&mut ea, &data)[0];

    println!("res={}", ea[res].data);
    println!("loss={loss}");
}

#[test]
fn test_mlp_training_x_mul_y() {
    for _ in 0..50 {
        let mut winner_precision = f64::MAX;
        let mut winner = vec![];

        for layers in 1..5 {
            for neurons in 3..10 {
                let mut ea = ExprArena::default();
                let mut config = vec![neurons; layers];
                config.push(1);

                let mlp = MultiLayerPerceptron::new(&mut ea, 2, &config);

                let mut training_data = Vec::new();
                for x in 0..20 {
                    for y in 0..20 {
                        let x = x / 1_000;
                        let y = y / 1_000;
                        training_data.push((vec![x as f64, y as f64], (x * y) as f64));
                    }
                }

                let gradient_increment = 0.001;
                let loss_limit = 0.00001;
                let count_limit = 10000;
                let loss = mlp.train(
                    &mut ea,
                    training_data,
                    gradient_increment,
                    loss_limit,
                    count_limit,
                );
                assert!(loss < loss_limit);

                let mut sum = 0.0;
                let mut count = 0;
                for x in 30..50 {
                    for y in 30..50 {
                        let x = 30 as f64 / 1_000.0;
                        let y = 30 as f64 / 1_000.0;
                        let expected = x * y;

                        let data = ea.va([x, y]);
                        let got = mlp.forward(&mut ea, &data)[0];
                        let got = ea[got].data;

                        sum += (expected - got).powi(2);
                        count += 1;
                    }
                }
                let precision = (sum / (count as f64)).sqrt();

                if precision < winner_precision {
                    winner_precision = precision;
                    winner = config.clone()
                }
            }
        }

        println!("Round winner = {winner:?}  precision={winner_precision}");
    }
}
