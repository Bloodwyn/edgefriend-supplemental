#include <rapidobj/rapidobj.hpp>
#include <glm/glm.hpp> 
#include <edgefriend.h>
#include <iostream>
#include <span>
#include <fstream>
#include <ranges>

int main(int argc, char** argv) {
	std::cout << argv[0] << '\n';
	
	if (argc < 2) {
		std::cerr << "Missing file argument\n";
		return 1;
	}
	const char* file = argv[1];

	auto model = rapidobj::ParseFile(file);
	if (model.error) {
		std::cerr << "Error: OBJ file" << file << "could not be loaded.";
		return 2;
	}

	if (model.shapes.size() == 0) {
		std::cerr << "Error: OBJ file" << file << "does not contain a mesh.";
		return 3;
	}

	const auto& objmesh = model.shapes.front().mesh;
	if (model.shapes.size() != 1) {
		std::cerr << "Warning: demo will only process the first shape/object!";
	}

	auto positionSpan = std::span<glm::vec3>(
		reinterpret_cast<glm::vec3*>(model.attributes.positions.data()), model.attributes.positions.size() / 3);

	std::vector<glm::vec3> positions(positionSpan.begin(), positionSpan.end());

	std::vector<std::int32_t> indices;
	std::transform(objmesh.indices.begin(), objmesh.indices.end(), std::back_inserter(indices),
		[](const rapidobj::Index& idx) {
		return idx.position_index;
	});

	std::vector<std::int32_t> indicesOffsets;

	indicesOffsets.reserve(objmesh.num_face_vertices.size());
	std::size_t startIndex = 0;
	for (const auto faceSize : objmesh.num_face_vertices) {
		indicesOffsets.push_back(startIndex);
		startIndex += faceSize;
	}

	ankerl::unordered_dense::map<glm::ivec2, float> creases;
	creases.reserve(objmesh.creases.size());
	for (const auto& crease : objmesh.creases) {
		const auto [min, max] = std::minmax(crease.position_index_from, crease.position_index_to);
		creases.emplace(glm::ivec2(min, max), crease.sharpness);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	auto edgefriendGeometry = SubdivideToEdgefriendGeometry(positions, indices, indicesOffsets, creases);
	auto endTime = std::chrono::high_resolution_clock::now();	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);	
	std::cout << "Preprocessing time: " << duration.count()/1000.f << "ms\n";
	
	for (int i = 0; i < 3; ++i) {
		edgefriendGeometry = SubdivideEdgefriendGeometry(edgefriendGeometry);
	}

	std::ofstream obj("output.obj");
	for (const auto& position : edgefriendGeometry.positions) {
		obj << "v " << position.x << ' ' << position.y << ' ' << position.z << '\n';
	}

	for (int i = 0; i < edgefriendGeometry.friendsAndSharpnesses.size(); ++i) {
		obj << 'f';
		for (int j = 0; j < 4; ++j) {
			obj << ' ' << edgefriendGeometry.indices[4 * i + j] + 1;
		}
		obj << '\n';
	}
	obj.close();

	return 0;
}