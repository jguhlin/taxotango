use burn::data::dataset::{Dataset, DatasetIterator};
use petgraph::algo::astar;
use petgraph::prelude::*;
// Uses too much memory
// use petgraph::algo::floyd_warshall;
use burn::data::dataset::SqliteDatasetStorage;
use crossbeam::channel::bounded;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::Arc;

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

pub fn build_taxonomy_graph_generator(nodes_file: &str, names_file: &str) -> BatchGenerator {
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
    // let graph = UnGraph::<u32, (), u32>::from_edges(&edges);
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

    let mut count = 0;

    let mut idx = 0;
    let mut idx2 = 0;

    BatchGenerator {
        graph: Arc::new(graph),
        rng: Xoshiro256PlusPlus::seed_from_u64(1337),
        epoch_size: 1024,
        levels,
    }
}

pub struct BatchGenerator {
    graph: Arc<Graph<u32, (), Undirected, u32>>,
    // graph: MatrixGraph<u32, (), petgraph::Undirected, std::option::Option<()>, u32>,
    rng: Xoshiro256PlusPlus,
    epoch_size: usize,
    pub levels: HashMap<u32, TaxaLevel>,
}

impl Dataset<TaxaDistance> for BatchGenerator {
    fn len(&self) -> usize {
        // Batch size of 1024
        self.epoch_size
    }

    fn get(&self, _index: usize) -> Option<TaxaDistance> {
        let mut rng = thread_rng();

        let len = self.graph.node_count();

        let i = rng.gen_range(0..len) as u32;

        // Let's random walk to find 16 nearby nodes
        // Then 16 far away nodes

        // Nearby <= 6
        // Distance >= 20 (max is 54, I think)
        let idx = self.graph.node_indices().nth(i as usize).unwrap();

        let mut branches = [0; 2];
        let mut distances = [0; 2];

        //for i in 0..20 {
        let depth = rng.gen_range(1..8);
        let (node, depth) = self.random_walk(idx, depth);
        branches[0] = node.index() as u32;
        distances[0] = depth as u32;
        // }

        let depth = rng.gen_range(10..40);
        let (node, depth) = self.random_walk(idx, depth);
        branches[1] = node.index() as u32;
        distances[1] = depth as u32;

        Some(TaxaDistance {
            origin: idx.index() as u32,
            branches,
            distances,
        })
    }

    // Provided methods
    fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
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
            epoch_size: batch_size,
            levels,
        }
    }

    pub fn random_walk(&self, start: NodeIndex, depth: u8) -> (NodeIndex, u8) {
        // Find a node at depth from the current start
        let mut rng = self.rng.clone();
        let mut current = start;
        let mut current_depth = 0;

        while current_depth < depth {
            let neighbors = self.graph.neighbors(current).collect::<Vec<NodeIndex>>();
            let next = neighbors[rng.gen_range(0..neighbors.len())];
            current = next;
            current_depth += 1;
        }

        // Get actual distance with astar

        let distance = astar(
            Arc::as_ref(&self.graph),
            start,
            |finish| finish == current,
            |e| 1,
            |_| 0,
        );

        let distance = distance.unwrap().0 as u8;

        (current, distance)
    }

    pub fn taxonomy_size(&self) -> usize {
        self.graph.node_count()
    }

    pub fn train(&self) -> Self {
        log::info!("Cloning BatchGenerator for training");

        let mut rng = self.rng.clone();
        rng.long_jump();

        Self {
            graph: Arc::clone(&self.graph),
            rng,
            levels: self.levels.clone(),
            epoch_size: 1024 * 512,
        }
    }

    pub fn test(&self) -> Self {
        // Limit data
        Self {
            graph: Arc::clone(&self.graph),
            rng: self.rng.clone(),
            levels: self.levels.clone(),
            epoch_size: 1024,
        }
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
