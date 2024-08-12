use burn::backend::Wgpu;
use burn::data::dataset::{Dataset, DatasetIterator};
use burn::prelude::*;
use petgraph::algo::astar;
use petgraph::graph::Node;
use petgraph::matrix_graph::{MatrixGraph, NotZero, UnMatrix};
use petgraph::prelude::*;
// Uses too much memory
// use petgraph::algo::floyd_warshall;
use burn::data::dataset::SqliteDatasetStorage;
use crossbeam::channel::bounded;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::Shutdown;
use std::sync::{Arc, atomic::AtomicBool};
use std::{cmp::min};

pub mod model;
pub use model::*;

pub enum TaxonomyWriterMessage {
    Write(Vec<(NodeIndex, NodeIndex, u8)>),
    Completed,
}

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy)]
pub enum TaxaLevel {
    NoRank,
    Root,
    Superkingdom,
    Kingdom,
    Subkingdom,
    Superphylum,
    Phylum,
    Subphylum,
    Superclass,
    Class,
    Infraclass,
    Subclass,
    Superorder,
    Order,
    Parvorder,
    Infraorder,
    Suborder,
    Superfamily,
    Family,
    Subfamily,
    Tribe,
    Subtribe,
    Genus,
    Subgenus,
    SpeciesGroup,
    Species,
    Subspecies,
    Clade,
    Forma,
    Varietas,
    SpeciesSubgroup,
    Subcohort,
    Cohort,
    Section,
    Subsection,
    Series,
}

impl TaxaLevel {
    pub fn from_str(rank: &str) -> Self {
        match rank {
            "no rank" => Self::NoRank,
            "species" => Self::Species,
            "tribe" => Self::Tribe,
            "genus" => Self::Genus,
            "superfamily" => Self::Superfamily,
            "family" => Self::Family,
            "subfamily" => Self::Subfamily,
            "order" => Self::Order,
            "infraorder" => Self::Infraorder,
            "suborder" => Self::Suborder,
            "class" => Self::Class,
            "subclass" => Self::Subclass,
            "phylum" => Self::Phylum,
            "subphylum" => Self::Subphylum,
            "kingdom" => Self::Kingdom,
            "superkingdom" => Self::Superkingdom,
            "root" => Self::Root,
            "subspecies" => Self::Subspecies,
            "species group" => Self::SpeciesGroup,
            "subgenus" => Self::Subgenus,
            "clade" => Self::Clade,
            "forma" => Self::Forma,
            "varietas" => Self::Varietas,
            "infraclass" => Self::Infraclass,
            "superorder" => Self::Superorder,
            "superclass" => Self::Superclass,
            "parvorder" => Self::Parvorder,
            "species subgroup" => Self::SpeciesSubgroup,
            "subcohort" => Self::Subcohort,
            "cohort" => Self::Cohort,
            "subtribe" => Self::Subtribe,
            "section" => Self::Section,
            "series" => Self::Series,
            "subkingdom" => Self::Subkingdom,
            "superphylum" => Self::Superphylum,
            "subsection" => Self::Subsection,
            _ => panic!("Unknown rank: {}", rank),
        }
    }
}


pub fn build_taxonomy_graph_limit_depth_csv_output(nodes_file: &str, names_file: &str, depth: u8) {
    // acc2tax todo: change it so we are passing &str
    let names = parse_names(names_file.to_string());
    let nodes = parse_nodes(nodes_file.to_string());

    log::info!("Parsed names: {} total", names.0.len());
    log::debug!("Parsed names (first 10): {:?}", &names.0[0..10]);
    log::debug!("Parsed names (u32, first 10): {:?}", &names.1[0..10]);
    log::info!("Parsed nodes: {} total", nodes.0.len());
    log::debug!("Parsed nodes (first 10): {:?}", &nodes.0[0..10]);
    log::debug!("Parsed nodes (Strings, first 10): {:?}", &nodes.1[0..10]);

    // Put names into a hashmap
    let names = names
        .0
        .iter()
        .enumerate()
        .map(|(idx, name)| (idx as u32, name.clone()))
        .collect::<HashMap<u32, String>>();

    // When a parent is 0 it means drop it, 1 is the root
    let edges: Vec<(u32, u32)> = nodes.0;
    println!("Length: {}", edges.len());
    let edges = edges
        .into_iter()
        .filter(|(x, y)| *x != 0 && *y != 0)
        .collect::<Vec<(u32, u32)>>();
    println!("Length: {}", edges.len());

    let levels: HashMap<u32, TaxaLevel> = nodes
        .1
        .iter()
        .enumerate()
        .map(|(idx, rank)| (idx as u32, TaxaLevel::from_str(rank)))
        .collect();

    // Filter for distinct, pull from both x and y
    let nodes: Vec<u32> = edges.iter().flat_map(|(x, y)| [x, y]).copied().collect();
    let nodes = nodes.into_iter().collect::<HashSet<u32>>();
    let nodes = nodes.into_iter().collect::<Vec<u32>>();

    assert!(nodes.len() <= u32::MAX as usize);

    // Dev, limit to first 1000 nodes
    // let nodes = nodes.iter().take(1000).copied().collect::<Vec<u32>>();

    log::debug!("Total Edges: {}", edges.len());

    log::info!("Building taxonomy graph");
    // let mut graph = UnGraph::<u32, (), u32>::from_edges(&edges);
    // let graph = MatrixGraph::<u32, (), Undirected, Option<()>, u32>::from_edges(&edges);

    let mut graph = UnGraph::<u32, (), u32>::with_capacity(nodes.len(), nodes.len());

    let nodes: HashMap<u32, NodeIndex> = nodes.iter().map(|x| (*x, graph.add_node(*x))).collect();

    for (x, y) in edges {
        graph.add_edge(nodes[&x], nodes[&y], ());
    }

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    // For all nodes, calculate all distances for a depth of _depth_
    // for i in graph.node_indices() {

    let (sender, receiver) = bounded::<TaxonomyWriterMessage>(8192);

    let is_finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let shutdown = std::sync::Arc::clone(&is_finished);
    let graph2 = graph.clone();

    let mut output_file = File::create("/mnt/data/data/taxonomy.db/taxonomy.csv.zst").expect("Error creating file");
    let mut output_file = BufWriter::with_capacity(8 * 1024, output_file);
    let mut output_file = zstd::stream::Encoder::new(output_file, 3).expect("Error creating zstd encoder");
    output_file.set_parameter(zstd::stream::raw::CParameter::NbWorkers(2)).expect("Error setting compression level");

    let csv_writer_join_handle = std::thread::spawn(move || {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
        let graph = graph2;

        let mut total_items: usize = 0;

        let total_size = graph.node_count();
        let pb = ProgressBar::new(total_size as u64);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {eta} {msg}")
                .unwrap(),
        );

        let mut max_depth = 0;
        let mut max_connections = 0;

        while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            match receiver.recv() {
                Ok(TaxonomyWriterMessage::Write(queue)) => {

                    if queue.len() > max_connections {
                        max_connections = queue.len();
                    }

                    pb.inc(1);
                    pb.set_message(format!("Total: {} / Current Queue: {} / Max Depth: {} / Channel Len: {} / Max Connections: {}", total_items, queue.len(), max_depth, receiver.len(), max_connections));

                    queue.into_iter().map(|(x, y, z)| {
                        if max_depth < z {
                          max_depth = z;
                        }

                        if x == y {
                            return None
                        }

                        // If depth > 1, randomly keep
                        if z > 1 && rng.gen_bool(0.4 / z as f64) {
                            return None
                        }

                        let bin = match rng.gen_bool(0.0005) {
                            true => "Test",
                            false => "Train",
                        };

                        total_items += 1;

                        // writeln!(output_file, "{},{},{},{}", graph[x], graph[y], z, bin).expect("Error writing to file");
                        Some(format!("{},{},{},{}\n", graph[x], graph[y], z, bin))
                    })
                    .filter_map(|x| x)                    
                    .for_each(|x| {
                        output_file.write_all(x.as_bytes()).expect("Error writing to file");
                    });

                    /*

                    for (x, y, z) in queue {
                        if max_depth < z {
                          max_depth = z;
                        }

                        // If depth > 1, randomly keep
                        if z > 1 && rng.gen_bool(1.0 / z as f64) {
                            continue;
                        }

                        let bin = match rng.gen_bool(0.0005) {
                            true => "Test",
                            false => "Train",
                        };

                        writeln!(output_file, "{},{},{},{}", graph[x], graph[y], z, bin).expect("Error writing to file");

                        total_items += 1;
                    } */
                }
                Ok(TaxonomyWriterMessage::Completed) => {
                    // pb.inc(1);
                    // deprecated
                }
                Err(_) => {}
            }
        }
        pb.finish_with_message(format!("Finished writing to database. Total items: {}", total_items));
    });

    for i in graph.node_indices() {
        if is_finished.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let mut cur_depth = 0;

        let mut queue = vec![(i, cur_depth)];
        let neighbors = graph.neighbors(i.into()).collect::<Vec<_>>();
        let mut used = HashSet::new();

        neighbors.into_iter().for_each(|x| {
            if !used.contains(&x) {
                used.insert(x);
                queue.push((x, cur_depth + 1));
            }
        });

        cur_depth += 1;

        let mut working_queue: Vec<(NodeIndex, u8)> = queue.clone();
        let mut new_queue = Vec::new();

        while cur_depth < depth {
            for (node_idx, node_depth) in working_queue.iter() {
                // Only on nodes at the current depth do we extract more neighbors
                let neighbors = graph.neighbors((*node_idx).into()).collect::<Vec<_>>();
                neighbors.into_iter().for_each(|x| {
                    if !used.contains(&x) {
                        used.insert(x);
                        new_queue.push((x, cur_depth + 1));
                    }
                });
            }

            working_queue = new_queue.clone();
            queue.extend(new_queue.clone());
            new_queue.clear();
            cur_depth += 1;
        }

        let queue: Vec<(NodeIndex, NodeIndex, u8)> =
            queue.iter().map(|(x, y)| (i, *x, *y)).collect();

        /*
        if queue.len() >= 1000 {
            // Print name of query node
            let name = names.get(&graph[i]).unwrap();
            println!(
                "Query Taxaid: {} {:?} {} - Query Node: {} - Queue Size: {} - Max Depth: {}",
                graph[i],
                i,
                i.index(),
                name,
                queue.len(),
                queue.iter().map(|x| x.2).max().unwrap()
            );
        } */

        let mut result = sender.try_send(TaxonomyWriterMessage::Write(queue));

        while let Err(msg) = result {
            match msg {
                crossbeam::channel::TrySendError::Full(msg) => {
                    std::thread::sleep(std::time::Duration::from_millis(270));
                    result = sender.try_send(msg);
                }
                crossbeam::channel::TrySendError::Disconnected(_) => {
                    is_finished.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }
        }

        graph.remove_node(i).expect("Error removing node");
    } 

    /*
    let node_indices = graph.node_indices().collect::<Vec<_>>();
    let indices: Vec<usize> = (0..node_indices.len()).collect::<Vec<_>>();

    indices.par_iter().for_each(|i| {
        if is_finished.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let i = node_indices[*i];
        let mut cur_depth = 0;

        let mut queue = vec![(i, cur_depth)];
        let mut used = HashSet::new();
        let neighbors = graph.neighbors(i.into()).collect::<Vec<_>>();
        neighbors.into_iter().for_each(|x| {
            if !used.contains(&x) {
                used.insert(x);
                queue.push((x, cur_depth + 1));
            }
        });

        cur_depth += 1;

        let mut working_queue: Vec<(NodeIndex, u8)> = queue.clone();
        let mut new_queue = Vec::new();

        while cur_depth < depth {

            for (node_idx, node_depth) in working_queue.iter() {
                // Depth we are working on
                if *node_depth < cur_depth {
                    continue
                }

                // Only on nodes at the current depth do we extract more neighbors
                let neighbors = graph.neighbors((*node_idx).into()).collect::<Vec<_>>();
                neighbors.into_iter().for_each(|x| {
                    if !used.contains(&x) {
                        used.insert(x);
                        new_queue.push((x, cur_depth + 1));
                    }
                });
            }

            working_queue = new_queue.clone();
            queue.extend(new_queue.clone());
            new_queue.clear();
            cur_depth += 1;
        }

        let queue: Vec<(NodeIndex, NodeIndex, u8)> = queue
            .iter()
            .map(|(x, y)| (i, *x, *y))
            .collect();

        let mut result = sender.try_send(TaxonomyWriterMessage::Write(queue));

        while let Err(msg) = result {
            match msg {
                crossbeam::channel::TrySendError::Full(msg) => {
                    std::thread::sleep(std::time::Duration::from_millis(270));
                    result = sender.try_send(msg);
                },
                crossbeam::channel::TrySendError::Disconnected(_) => {
                    is_finished.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }

        }
    });
    */

    is_finished.store(true, std::sync::atomic::Ordering::Relaxed);

    csv_writer_join_handle.join().unwrap();

    log::info!("Finished writing to database");
}

pub fn build_taxonomy_graph_limit_depth(nodes_file: &str, names_file: &str, depth: u8) {
    // acc2tax todo: change it so we are passing &str
    let names = parse_names(names_file.to_string());
    let nodes = parse_nodes(nodes_file.to_string());

    log::info!("Parsed names: {} total", names.0.len());
    log::debug!("Parsed names (first 10): {:?}", &names.0[0..10]);
    log::debug!("Parsed names (u32, first 10): {:?}", &names.1[0..10]);
    log::info!("Parsed nodes: {} total", nodes.0.len());
    log::debug!("Parsed nodes (first 10): {:?}", &nodes.0[0..10]);
    log::debug!("Parsed nodes (Strings, first 10): {:?}", &nodes.1[0..10]);

    // Put names into a hashmap
    let names = names
        .0
        .iter()
        .enumerate()
        .map(|(idx, name)| (idx as u32, name.clone()))
        .collect::<HashMap<u32, String>>();

    // When a parent is 0 it means drop it, 1 is the root
    let edges: Vec<(u32, u32)> = nodes.0;
    println!("Length: {}", edges.len());
    let edges = edges
        .into_iter()
        .filter(|(x, y)| *x != 0 && *y != 0)
        .collect::<Vec<(u32, u32)>>();
    println!("Length: {}", edges.len());

    let levels: HashMap<u32, TaxaLevel> = nodes
        .1
        .iter()
        .enumerate()
        .map(|(idx, rank)| (idx as u32, TaxaLevel::from_str(rank)))
        .collect();

    // Filter for distinct, pull from both x and y
    let nodes: Vec<u32> = edges.iter().flat_map(|(x, y)| [x, y]).copied().collect();
    let nodes = nodes.into_iter().collect::<HashSet<u32>>();
    let nodes = nodes.into_iter().collect::<Vec<u32>>();

    assert!(nodes.len() <= u32::MAX as usize);

    // Dev, limit to first 1000 nodes
    // let nodes = nodes.iter().take(1000).copied().collect::<Vec<u32>>();

    log::debug!("Total Edges: {}", edges.len());

    log::info!("Building taxonomy graph");
    // let mut graph = UnGraph::<u32, (), u32>::from_edges(&edges);
    // let graph = MatrixGraph::<u32, (), Undirected, Option<()>, u32>::from_edges(&edges);

    let mut graph = UnGraph::<u32, (), u32>::with_capacity(nodes.len(), nodes.len());

    let nodes: HashMap<u32, NodeIndex> = nodes.iter().map(|x| (*x, graph.add_node(*x))).collect();

    for (x, y) in edges {
        graph.add_edge(nodes[&x], nodes[&y], ());
    }

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    // Store everything into sql database
    let storage =
        SqliteDatasetStorage::from_name("TaxonomyDB").with_base_dir("/mnt/data/data/taxonomy.db");

    let mut writer = storage.writer::<TaxaDistance>(true).unwrap();

    // For all nodes, calculate all distances for a depth of _depth_
    // for i in graph.node_indices() {

    let (sender, receiver) = bounded::<TaxonomyWriterMessage>(8192);

    let is_finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let shutdown = std::sync::Arc::clone(&is_finished);
    let graph2 = graph.clone();

    let db_writer_join_handle = std::thread::spawn(move || {
        // let mut already_inserted = HashSet::with_capacity(nodes.len() * 10);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);
        let graph = graph2;

        let mut total_items = 0;

        let total_size = graph.node_count();
        let pb = ProgressBar::new(total_size as u64);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {eta} {msg}")
                .unwrap(),
        );

        let mut max_depth = 0;
        let mut max_connections = 0;

        while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            match receiver.recv() {
                Ok(TaxonomyWriterMessage::Write(queue)) => {

                    if queue.len() > max_connections {
                        max_connections = queue.len();
                    }

                    pb.inc(1);
                    pb.set_message(format!("Total: {} / Current Queue: {} / Max Depth: {} / Channel Len: {} / Max Connections: {}", total_items, queue.len(), max_depth, receiver.len(), max_connections));

                    for (x, y, z) in queue {
                        if max_depth < z {
                          max_depth = z;
                        }



                        // If depth > 1, randomly keep
                        if z > 1 && rng.gen_bool(1.0 / z as f64) {
                            continue;
                        }

                        // If already inserted, skip
                        // Sort the keys
                        //let mut key = [x.index(), y.index()];
                        // key.sort_unstable();

                        // if already_inserted.contains(&key) {
                            //continue;
                        //}

                        // already_inserted.insert(key);

                        let bin = match rng.gen_bool(0.0005) {
                            true => "Test",
                            false => "Train",
                        };

                        writer
                            .write(
                                bin,
                                &TaxaDistance {
                                    branches: [graph[x], graph[y]],
                                    distance: z as f32,
                                },
                            )
                            .expect("Error writing to database");

                        total_items += 1;
                    }
                }
                Ok(TaxonomyWriterMessage::Completed) => {
                    // pb.inc(1);
                    // deprecated
                }
                Err(_) => {}
            }
        }
        writer.set_completed().expect("Error setting complete");
        pb.finish_with_message(format!("Finished writing to database. Total items: {}", total_items));
    });

    for i in graph.node_indices() {
        if is_finished.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let mut cur_depth = 0;

        let mut queue = vec![(i, cur_depth)];
        let neighbors = graph.neighbors(i.into()).collect::<Vec<_>>();
        let mut used = HashSet::new();

        neighbors.into_iter().for_each(|x| {
            if !used.contains(&x) {
                used.insert(x);
                queue.push((x, cur_depth + 1));
            }
        });

        cur_depth += 1;

        let mut working_queue: Vec<(NodeIndex, u8)> = queue.clone();
        let mut new_queue = Vec::new();

        while cur_depth < depth {
            for (node_idx, node_depth) in working_queue.iter() {
                // Only on nodes at the current depth do we extract more neighbors
                let neighbors = graph.neighbors((*node_idx).into()).collect::<Vec<_>>();
                neighbors.into_iter().for_each(|x| {
                    if !used.contains(&x) {
                        used.insert(x);
                        new_queue.push((x, cur_depth + 1));
                    }
                });
            }

            working_queue = new_queue.clone();
            queue.extend(new_queue.clone());
            new_queue.clear();
            cur_depth += 1;
        }

        let queue: Vec<(NodeIndex, NodeIndex, u8)> =
            queue.iter().map(|(x, y)| (i, *x, *y)).collect();

        /*
        if queue.len() >= 1000 {
            // Print name of query node
            let name = names.get(&graph[i]).unwrap();
            println!(
                "Query Taxaid: {} {:?} {} - Query Node: {} - Queue Size: {} - Max Depth: {}",
                graph[i],
                i,
                i.index(),
                name,
                queue.len(),
                queue.iter().map(|x| x.2).max().unwrap()
            );
        } */

        let mut result = sender.try_send(TaxonomyWriterMessage::Write(queue));

        while let Err(msg) = result {
            match msg {
                crossbeam::channel::TrySendError::Full(msg) => {
                    std::thread::sleep(std::time::Duration::from_millis(270));
                    result = sender.try_send(msg);
                }
                crossbeam::channel::TrySendError::Disconnected(_) => {
                    is_finished.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }
        }

        graph.remove_node(i).expect("Error removing node");
    } 

    /*
    let node_indices = graph.node_indices().collect::<Vec<_>>();
    let indices: Vec<usize> = (0..node_indices.len()).collect::<Vec<_>>();

    indices.par_iter().for_each(|i| {
        if is_finished.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let i = node_indices[*i];
        let mut cur_depth = 0;

        let mut queue = vec![(i, cur_depth)];
        let mut used = HashSet::new();
        let neighbors = graph.neighbors(i.into()).collect::<Vec<_>>();
        neighbors.into_iter().for_each(|x| {
            if !used.contains(&x) {
                used.insert(x);
                queue.push((x, cur_depth + 1));
            }
        });

        cur_depth += 1;

        let mut working_queue: Vec<(NodeIndex, u8)> = queue.clone();
        let mut new_queue = Vec::new();

        while cur_depth < depth {

            for (node_idx, node_depth) in working_queue.iter() {
                // Depth we are working on
                if *node_depth < cur_depth {
                    continue
                }

                // Only on nodes at the current depth do we extract more neighbors
                let neighbors = graph.neighbors((*node_idx).into()).collect::<Vec<_>>();
                neighbors.into_iter().for_each(|x| {
                    if !used.contains(&x) {
                        used.insert(x);
                        new_queue.push((x, cur_depth + 1));
                    }
                });
            }

            working_queue = new_queue.clone();
            queue.extend(new_queue.clone());
            new_queue.clear();
            cur_depth += 1;
        }

        let queue: Vec<(NodeIndex, NodeIndex, u8)> = queue
            .iter()
            .map(|(x, y)| (i, *x, *y))
            .collect();

        let mut result = sender.try_send(TaxonomyWriterMessage::Write(queue));

        while let Err(msg) = result {
            match msg {
                crossbeam::channel::TrySendError::Full(msg) => {
                    std::thread::sleep(std::time::Duration::from_millis(270));
                    result = sender.try_send(msg);
                },
                crossbeam::channel::TrySendError::Disconnected(_) => {
                    is_finished.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }

        }
    });
    */

    is_finished.store(true, std::sync::atomic::Ordering::Relaxed);

    db_writer_join_handle.join().unwrap();

    log::info!("Finished writing to database");
}

pub fn build_taxonomy_graph_generator(nodes_file: &str, names_file: &str) {
    // acc2tax todo: change it so we are passing &str
    let names = parse_names(names_file.to_string());
    let nodes = parse_nodes(nodes_file.to_string());

    log::info!("Parsed names: {} total", names.0.len());
    log::debug!("Parsed names (first 10): {:?}", &names.0[0..10]);
    log::debug!("Parsed names (u32, first 10): {:?}", &names.1[0..10]);
    log::info!("Parsed nodes: {} total", nodes.0.len());
    log::debug!("Parsed nodes (first 10): {:?}", &nodes.0[0..10]);
    log::debug!("Parsed nodes (Strings, first 10): {:?}", &nodes.1[0..10]);

    // When a parent is 0 it means drop it, 1 is the root
    let edges: Vec<(u32, u32)> = nodes.0;

    let levels: HashMap<u32, TaxaLevel> = nodes
        .1
        .iter()
        .enumerate()
        .map(|(idx, rank)| (idx as u32, TaxaLevel::from_str(rank)))
        .collect();

    // Filter for distinct, pull from both x and y
    let nodes: Vec<u32> = edges.iter().flat_map(|(x, y)| [x, y]).copied().collect();
    let nodes = nodes.into_iter().collect::<HashSet<u32>>();
    let nodes = nodes.into_iter().collect::<Vec<u32>>();

    // Dev, limit to first 1000 nodes
    // let nodes = nodes.iter().take(1000).copied().collect::<Vec<u32>>();

    log::debug!("Total Edges: {}", edges.len());

    log::info!("Building taxonomy graph");
    let graph = UnGraph::<u32, (), u32>::from_edges(&edges);
    // let graph = MatrixGraph::<u32, (), Undirected, Option<()>, u32>::from_edges(&edges);

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    // Store everything into sql database
    let storage =
        SqliteDatasetStorage::from_name("TaxonomyDB").with_base_dir("/mnt/data/data/taxonomy.db");

    let mut writer = storage.writer::<TaxaDistance>(true).unwrap();

    let mut count = 0;

    let mut idx = 0;
    let mut idx2 = 0;

    let total = nodes.len() * nodes.len();

    let mut pairs = Vec::new();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1337);

    // Do in batches of 1024
    let mut start = 0;
    let mut longest_distance = 0;
    for end in (0..total)
        .step_by(1024)
        .skip(1)
        .chain(std::iter::once(total))
    {
        for i in start..end {
            let idx = nodes[i / nodes.len()];
            let idx2 = nodes[i % nodes.len()];

            pairs.push((idx, idx2));
        }

        let batch: Vec<(u32, u32, usize)> = pairs
            .par_iter()
            .map(|(idx, idx2)| {
                let dist = astar(
                    &graph,
                    (*idx).into(),
                    |finish| finish == (*idx2).into(),
                    |_| 1,
                    |_| 0,
                )
                .map(|x| x.1)
                .expect(format!("No path found - {} - {}", idx, idx2).as_str())
                .len();
                (idx.clone(), idx2.clone(), dist)
            })
            .collect();

        if longest_distance < batch.iter().map(|x| x.2).max().unwrap() {
            longest_distance = batch.iter().map(|x| x.2).max().unwrap();
        }

        batch.iter().for_each(|(x, y, z)| {
            let bin = match rng.gen_bool(0.0005) {
                true => "Test",
                false => "Train",
            };
            writer
                .write(
                    bin,
                    &TaxaDistance {
                        branches: [*x, *y],
                        distance: *z as f32,
                    },
                )
                .expect("Error writing to database");
        });

        count += 1;

        println!(
            "Batch: {} - {}/{} - Longest Distance: {}",
            count,
            count * 1024,
            total,
            longest_distance
        );
        start = end;

        pairs.clear();
    }

    log::info!("Longest distance: {}", longest_distance);
    writer.set_completed().expect("Error setting complete");
}

pub struct BatchGenerator {
    graph: Arc<Graph<u32, (), Undirected, u32>>,
    // graph: MatrixGraph<u32, (), petgraph::Undirected, std::option::Option<()>, u32>,
    rng: Xoshiro256PlusPlus,
    batch_size: usize,
    pub nodes: Vec<u32>,
    pub levels: HashMap<u32, TaxaLevel>,
}

impl Dataset<TaxaDistance> for BatchGenerator {
    fn len(&self) -> usize {
        self.nodes.len() * self.nodes.len()
    }

    fn get(&self, index: usize) -> Option<TaxaDistance> {
        let idx = index / self.nodes.len();
        let idx2 = index % self.nodes.len();

        let idx = self.nodes[idx];
        let idx2 = self.nodes[idx2];

        let dist = astar(
            Arc::as_ref(&self.graph),
            idx.into(),
            |finish| finish == idx2.into(),
            |_| 1,
            |_| 0,
        )
        .map(|x| x.1)
        .expect(format!("No path found - {} - {}", idx, idx2).as_str())
        .len();

        Some(TaxaDistance {
            branches: [idx, idx2],
            distance: dist as f32,
        })
    }

    // Provided methods
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn iter(&self) -> DatasetIterator<'_, TaxaDistance>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

impl BatchGenerator {
    pub fn new(
        graph: Graph<u32, (), Undirected, u32>,
        rng: Xoshiro256PlusPlus,
        batch_size: usize,
        nodes: Vec<u32>,
        levels: HashMap<u32, TaxaLevel>,
    ) -> Self {
        Self {
            graph: Arc::new(graph),
            rng,
            batch_size,
            nodes,
            levels,
        }
    }

    pub fn train(&self) -> Self {
        log::info!("Cloning BatchGenerator for training");
        Self {
            graph: self.graph.clone(),
            rng: self.rng.clone(),
            batch_size: self.batch_size,
            nodes: self.nodes.clone(),
            levels: self.levels.clone(),
        }
    }

    pub fn test(&self) -> Self {
        // Limit data
        let nodes = self.nodes.iter().take(100).copied().collect::<Vec<u32>>();
        let levels = self
            .levels
            .iter()
            .filter(|(k, _)| nodes.contains(k))
            .map(|(k, v)| (*k, *v))
            .collect::<HashMap<u32, TaxaLevel>>();

        Self {
            graph: self.graph.clone(),
            rng: self.rng.clone(),
            batch_size: self.batch_size,
            nodes,
            levels,
        }
    }

    pub fn generate_batch(&mut self) -> Vec<(u32, u32, usize)> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            let idx = *self
                .nodes
                .choose(&mut self.rng)
                .expect("Error choosing random node");
            let idx2 = *self
                .nodes
                .choose(&mut self.rng)
                .expect("Error choosing random node");

            pairs.push((idx, idx2));
        }

        pairs
            .par_iter()
            .map(|(idx, idx2)| {
                let dist = astar(
                    Arc::as_ref(&self.graph),
                    (*idx).into(),
                    |finish| finish == (*idx2).into(),
                    |_| 1,
                    |_| 0,
                )
                .map(|x| x.1)
                .expect(format!("No path found - {} - {}", idx, idx2).as_str())
                .len();
                (idx.clone(), idx2.clone(), dist)
            })
            .collect_into_vec(&mut batch);

        batch
    }
}

pub fn parse_nodes(filename: String) -> (Vec<(u32, u32)>, Vec<String>) {
    let mut taxon_to_parent: Vec<(u32, u32)> = Vec::with_capacity(4_000_000);
    let mut taxon_rank: Vec<String> = Vec::with_capacity(4_000_000);

    let reader = BufReader::new(File::open(filename).expect("Unable to open taxonomy names file"));

    let lines = reader.lines();

    for line in lines {
        let split = line
            .expect("Error reading line")
            .split('|')
            .map(|x| x.trim().to_string())
            .collect::<Vec<String>>();

        let tax_id: u32 = split[0].parse().expect("Error converting to number");
        let parent_id: u32 = split[1].parse().expect("Error converting to number");
        let rank: &str = &split[2];

        taxon_to_parent.push((tax_id, parent_id));
        taxon_rank.push(rank.into());
    }

    taxon_to_parent.shrink_to_fit();
    taxon_rank.shrink_to_fit();

    (taxon_to_parent, taxon_rank)
}

pub fn parse_names(filename: String) -> (Vec<String>, Vec<u32>) {
    let mut names: Vec<String> = Vec::with_capacity(3_006_098);

    let reader = BufReader::new(File::open(filename).expect("Unable to open taxonomy names file"));

    let lines = reader.lines();
    let mut taxids = HashSet::new();

    for line in lines {
        let split = line
            .expect("Error reading line")
            .split('|')
            .map(|x| x.trim().to_string())
            .collect::<Vec<String>>();

        let id: usize = split[0].parse().expect("Error converting to number");
        let name: &str = &split[1];
        let class: &str = &split[3];

        match names.get(id) {
            None => {
                names.resize(id + 1, "".to_string());
                names[id] = name.into();
                taxids.insert(id as u32);
            }
            Some(_) => {
                if class == "scientific name" {
                    names[id] = name.into();
                }
            }
        };
    }

    let mut taxids = taxids.into_iter().collect::<Vec<u32>>();
    taxids.sort_unstable();

    (names, taxids)
}
