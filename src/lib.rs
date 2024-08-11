use burn::backend::Wgpu;
use burn::prelude::*;
use petgraph::algo::astar;
use petgraph::prelude::*;
// Uses too much memory
// use petgraph::algo::floyd_warshall;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

// Type alias for the backend to use.
type Backend = Wgpu;

pub mod model;
pub use model::*;

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug)]
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

pub fn build_taxonomy_graph(nodes_file: &str, names_file: &str) -> BatchGenerator {
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

    log::debug!("Total Edges: {}", edges.len());

    log::info!("Building taxonomy graph");
    let graph = UnGraph::<u32, (), u32>::from_edges(&edges);

    log::info!(
        "Taxonomy graph built. Node Count: {} - Edge Count: {}",
        graph.node_count(),
        graph.edge_count()
    );

    BatchGenerator::new(
        graph,
        Xoshiro256PlusPlus::seed_from_u64(1337),
        1024,
        nodes,
        levels,
    )
}

pub struct BatchGenerator {
    graph: Graph<u32, (), Undirected, u32>,
    rng: Xoshiro256PlusPlus,
    batch_size: usize,
    pub nodes: Vec<u32>,
    pub levels: HashMap<u32, TaxaLevel>,
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
            graph,
            rng,
            batch_size,
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
                    &self.graph,
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
