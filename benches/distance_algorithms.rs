use criterion::{black_box, criterion_group, criterion_main, Criterion};
use petgraph::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{BinaryHeap, HashSet};
use taxotangolib::*;

fn criterion_benchmark(c: &mut Criterion) {
    let nodes_file = "/mnt/data/data/nt/taxdmp/nodes.dmp";
    let names_file = "/mnt/data/data/nt/taxdmp/names.dmp";
    let (graph, node_indices, node_indices_per_taxa_level, taxa_names, levels, levels_in_order, nodes, colors) =
        build_taxonomy_graph(nodes_file, names_file);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

    let starting_node = nodes.get(&1).unwrap();

    let mut group = c.benchmark_group("Random Walk Depth 6 Benchmark");
    group.sample_size(100);

    group.bench_function("random_walk1 depth 6", |b| {
        b.iter(|| {
            random_walk1(
                black_box(&graph),
                &mut rng,
                black_box(starting_node.clone()),
                black_box(6),
                vec![],
            )
        })
    });

    // Reset rng
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

    group.bench_function("random_walk2 depth 6", |b| {
        b.iter(|| {
            random_walk2(
                black_box(&graph),
                &mut rng,
                black_box(starting_node.clone()),
                black_box(6),
                vec![],
            )
        })
    });

    // Reset rng
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

    group.bench_function("random_walk_alt depth 6", |b| {
        b.iter(|| {
            random_walk_alt(
                black_box(&graph),
                &mut rng,
                black_box(starting_node.clone()),
                black_box(6),
                &HashSet::new(),
            )
        })
    });

    group.finish();

    let mut group = c.benchmark_group("Random Walk Depth 8 Benchmark");
    group.sample_size(100);

    group.bench_function("random_walk1 depth 8", |b| {
        b.iter(|| {
            random_walk1(
                black_box(&graph),
                &mut rng,
                black_box(starting_node.clone()),
                black_box(8),
                vec![],
            )
        })
    });

    // Reset rng
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

    group.bench_function("random_walk2 depth 8", |b| {
        b.iter(|| {
            random_walk2(
                black_box(&graph),
                &mut rng,
                black_box(starting_node.clone()),
                black_box(8),
                vec![],
            )
        })
    });

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

    group.bench_function("random_walk_alt depth 8", |b| {
        b.iter(|| {
            random_walk_alt(
                black_box(&graph),
                &mut rng,
                black_box(starting_node.clone()),
                black_box(8),
                &HashSet::new(),
            )
        })
    });

    group.finish();
}

fn random_walk_alt<R: Rng>(
    graph: &Graph<u32, (), Directed, u32>,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: &HashSet<NodeIndex>,
) -> Option<(NodeIndex, u32)> {
    let mut current_node = start;
    let mut visited_nodes = HashSet::new();
    visited_nodes.insert(current_node);

    for current_depth in 0..depth {
        let neighbors: Vec<_> = graph
            .neighbors(current_node)
            .filter(|&n| !visited_nodes.contains(&n) && !excluded_nodes.contains(&n))
            .collect();

        if neighbors.is_empty() {
            return if current_node != start {
                Some((current_node, current_depth as u32))
            } else {
                None
            };
        }

        let mut next_node_index = rng.gen_range(0..neighbors.len());

        // Try to avoid dead ends if not at the last step
        if current_depth < depth - 1 {
            let mut attempts = 0;
            while attempts < neighbors.len()
                && graph.neighbors(neighbors[next_node_index]).count() == 1
            {
                next_node_index = (next_node_index + 1) % neighbors.len();
                attempts += 1;
            }
        }

        current_node = neighbors[next_node_index];
        visited_nodes.insert(current_node);
    }

    Some((current_node, depth as u32))
}

fn random_walk1<R: Rng>(
    graph: &Graph<u32, (), Directed, u32>,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: Vec<NodeIndex>,
) -> Option<NodeIndex> {
    let mut current_node = start;
    let mut visited_nodes = vec![current_node];

    for curdepth in 0..depth {
        let mut neighbors: Vec<_> = graph
            .neighbors(current_node)
            .filter(|&n| !visited_nodes.contains(&n))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        neighbors = neighbors
            .into_iter()
            .filter(|x| !excluded_nodes.contains(x))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        let mut next_node = rng.gen_range(0..neighbors.len());
        while curdepth < depth - 1 && graph.neighbors(neighbors[next_node]).count() == 1 {
            if neighbors.len() == 1 {
                current_node = neighbors[next_node];
                if start != current_node {
                    return Some(current_node);
                } else {
                    return None;
                }
            }

            // Remove from neighbors
            neighbors.remove(next_node);

            next_node = rng.gen_range(0..neighbors.len());
        }

        visited_nodes.push(neighbors[next_node]);
        current_node = neighbors[next_node];
    }

    Some(current_node)
}

// Use binary heap for visited nodes...
fn random_walk2<R: Rng>(
    graph: &Graph<u32, (), Directed, u32>,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: Vec<NodeIndex>,
) -> Option<NodeIndex> {
    let mut current_node = start;
    let mut visited_nodes = BinaryHeap::new();
    visited_nodes.push(start);

    for curdepth in 0..depth {
        let mut neighbors: Vec<_> = graph
            .neighbors(current_node)
            .filter(|&n| !visited_nodes.iter().any(|x| *x == n))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        neighbors = neighbors
            .into_iter()
            .filter(|x| !excluded_nodes.contains(x))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        let mut next_node = rng.gen_range(0..neighbors.len());
        while curdepth < depth - 1 && graph.neighbors(neighbors[next_node]).count() == 1 {
            if neighbors.len() == 1 {
                current_node = neighbors[next_node];
                if start != current_node {
                    return Some(current_node);
                } else {
                    return None;
                }
            }

            // Remove from neighbors
            neighbors.remove(next_node);

            next_node = rng.gen_range(0..neighbors.len());
        }

        visited_nodes.push(neighbors[next_node]);
        current_node = neighbors[next_node];
    }

    Some(current_node)
}

// Binary heap didn't help, try something else...
fn random_walk3<R: Rng>(
    graph: &Graph<u32, (), Directed, u32>,
    rng: &mut R,
    start: NodeIndex,
    depth: usize,
    excluded_nodes: Vec<NodeIndex>,
) -> Option<NodeIndex> {
    let mut current_node = start;
    let mut visited_nodes = vec![current_node];

    for curdepth in 0..depth {
        let mut neighbors: Vec<_> = graph
            .neighbors(current_node)
            .filter(|&n| !visited_nodes.contains(&n))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        neighbors = neighbors
            .into_iter()
            .filter(|x| !excluded_nodes.contains(x))
            .collect();

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        if neighbors.is_empty() {
            if start != current_node {
                return Some(current_node);
            } else {
                return None;
            }
        }

        let mut next_node = rng.gen_range(0..neighbors.len());
        while curdepth < depth - 1 && graph.neighbors(neighbors[next_node]).count() == 1 {
            if neighbors.len() == 1 {
                current_node = neighbors[next_node];
                if start != current_node {
                    return Some(current_node);
                } else {
                    return None;
                }
            }

            // Remove from neighbors
            neighbors.remove(next_node);

            next_node = rng.gen_range(0..neighbors.len());
        }

        visited_nodes.push(neighbors[next_node]);
        current_node = neighbors[next_node];
    }

    Some(current_node)
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
