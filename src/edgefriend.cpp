#include <edgefriend.h>
#include <atomic>
#include <algorithm>
#include <execution>
#include <ranges>

#define EXECUTION_POLICY std::execution::par

#include <fstream>

EdgefriendGeometry SubdivideToEdgefriendGeometry(
	std::vector<glm::vec3> oldPositions,
	std::vector<int> oldIndices,
	std::vector<int> oldIndicesOffsets,
	ankerl::unordered_dense::map<glm::ivec2, float> oldCreases) {
	using Edge = glm::ivec2;
	struct EdgeSide {
		int face = -1;
		int corner = -1;
	};

	// --- define helper lambdas for later ---
	const auto UniqueEdge = [&](int a, int b) -> Edge {
		if (a < b) {
			return Edge(a, b);
		}
		return Edge(b, a);
	};

	const auto FaceSize = [&](int face) -> int {
		const int& curOffset = oldIndicesOffsets[face];
		const int nextOffset = (face + 1 < oldIndicesOffsets.size()) ? oldIndicesOffsets[face + 1] : oldIndices.size();
		return nextOffset - curOffset;
	};

	const auto Prev = [&](EdgeSide e) -> EdgeSide {
		int faceSize = FaceSize(e.face);
		e.corner = (e.corner + faceSize - 1) % faceSize;
		return e;
	};

	const auto Index = [&](int face, int corner) -> int {
		return oldIndices[oldIndicesOffsets[face] + corner];
	};

	const auto Sharpness = [&](Edge uniqueEdge) -> float {
		auto itr = oldCreases.find(uniqueEdge);
		if (itr == oldCreases.end()) {
			return 0.f;
		} else {
			return itr->second;
		}
	};

	// --- build hashmap to get neighbourhood information --- 
	// map an edge consisting of two vertex ids to an edge id and the two faces that are on each side of the edge
	//std::unordered_map<Edge, std::tuple<std::size_t, EdgeSide, EdgeSide>> edgeMap;
	// we recommend to use faster hashmap implementations...
	ankerl::unordered_dense::map<Edge, std::tuple<std::size_t, EdgeSide, EdgeSide>> edgeMap;

	for (int face = 0; face < oldIndicesOffsets.size(); ++face) {
		int faceSize = FaceSize(face);
		for (int corner = 0; corner < faceSize; ++corner) {
			const auto nextCorner = (corner + 1) % faceSize;

			int a = Index(face, corner);
			int b = Index(face, nextCorner);

			const auto uniqueEdge = UniqueEdge(a, b);

			const auto& [it, _] =
				edgeMap.emplace(uniqueEdge, std::make_tuple(edgeMap.size(), EdgeSide(), EdgeSide()));

			auto& [__, left, right] = it->second;

			if (uniqueEdge.x == a) {
				left.face = face;
				left.corner = corner;
			} else {
				right.face = face;
				right.corner = corner;
			}
		}
	}

	// --- close all borders ---
	// helper function for walking along the border
	const auto GetCCWTillBorder = [&](EdgeSide cur) {
		while (true) {
			auto prev = Prev(cur);
			const auto uniqueEdge = UniqueEdge(Index(cur.face, cur.corner), Index(prev.face, prev.corner));

			auto itr = edgeMap.find(uniqueEdge);
			auto& [_, a, b] = itr->second;

			EdgeSide newSide = (a.face == cur.face) ? b : a;

			if (newSide.face < 0) {
				return cur;
			}
			cur = newSide;
		}
	};

	// get a set of all bordered edges
	//std::unordered_set<Edge> borders;
	ankerl::unordered_dense::set<Edge> borders;
	for (const auto& [edge, entry] : edgeMap) {
		// check if one side of edge does not have a face
		if (std::get<1>(entry).face < 0 || std::get<2>(entry).face < 0) {
			borders.insert(edge);
		}
	}

	std::size_t nG = 0;

	std::vector<int> newFace;
	newFace.reserve(borders.size());

	while (!borders.empty()) {
		Edge border = *borders.begin();

		oldIndicesOffsets.push_back(oldIndices.size());

		auto& [_, a, b] = edgeMap.find(border)->second;
		bool aIsNew = (a.face == -1);

		EdgeSide start = aIsNew ? b : a;
		EdgeSide cur = start;
		do {
			newFace.push_back(oldIndices[oldIndicesOffsets[cur.face] + cur.corner]);
			cur = Prev(GetCCWTillBorder(cur));
		} while (cur.face != start.face || cur.corner != start.corner);

		nG += newFace.size();
		for (int i = 0; i < newFace.size(); ++i) {
			auto e = UniqueEdge(newFace[i], newFace[(i + 1) % newFace.size()]);
			borders.erase(e);
			auto& [_, a, b] = edgeMap.find(e)->second;
			auto& c = (a.face == -1) ? a : b;
			c.face = oldIndicesOffsets.size() - 1;
			c.corner = i;
		}
		oldIndices.insert(oldIndices.end(), newFace.begin(), newFace.end());
		newFace.clear();
	}

	// --- allocate buffers ---
	const std::size_t oV = oldPositions.size();
	const std::size_t oE = edgeMap.size();
	const std::size_t oF = oldIndicesOffsets.size();

	std::size_t nV = oV + oE + oF;
	std::size_t nF = oldIndices.size();
	std::size_t nC = 4 * nF;

	std::vector<glm::vec3>  newPositions(nV, glm::vec3(0, 0, 0));
	std::vector<int>        newIndices(4 * nF);
	std::vector<glm::uvec4> newFriendsAndSharpnesses(nF);
	std::vector<int>        newValenceStartInfos(nV);

	std::vector<std::atomic<EdgeSide>> vertexStart(oV);

	// --- compute face-points, topology, friends and valence start info ---
	auto faceView = std::views::iota(0ull, oF);
	std::for_each(EXECUTION_POLICY, faceView.begin(), faceView.end(), [&](int face) {
		auto       fp = oV + oE + face;
		int faceSize = FaceSize(face);
		for (int slot = 0; slot < faceSize; ++slot) {
			int v = Index(face, slot);

			vertexStart[v] ={face, slot};
			newPositions[fp] += oldPositions[v];

			int prev = Index(face, (slot + faceSize - 1) % faceSize);
			int next = Index(face, (slot + 1) % faceSize);

			const auto prevEdge = Edge(prev, v);
			const auto nextEdge = Edge(v, next);

			const auto prevEdgeUnique = UniqueEdge(prevEdge.x, prevEdge.y);
			const auto nextEdgeUnique = UniqueEdge(nextEdge.x, nextEdge.y);

			const auto& prevEdgeData = edgeMap.at(prevEdgeUnique);
			const auto& nextEdgeData = edgeMap.at(nextEdgeUnique);

			const auto& prevEdgeId = std::get<0>(prevEdgeData);
			const auto& nextEdgeId = std::get<0>(nextEdgeData);

			std::size_t cornerId = oldIndicesOffsets[face] + slot;

			newIndices[4 * cornerId + 0] = v;
			newIndices[4 * cornerId + 1] = oV + nextEdgeId;
			newIndices[4 * cornerId + 2] = fp;
			newIndices[4 * cornerId + 3] = oV + prevEdgeId;

			for (int i = 0; i < 4; ++i) {
				newValenceStartInfos[newIndices[4 * cornerId + i]] = 4 * cornerId + i;
			}

			auto friend0Face = oldIndicesOffsets[face] + ((slot + 1) % faceSize);
			auto friend0 = 2 * friend0Face + 1;

			auto [_, a, b] = prevEdgeData;
			a = (a.face == face) ? b : a;

			auto friend1Face = oldIndicesOffsets[a.face] + a.corner;
			auto friend1 = 2 * friend1Face + 0;

			newFriendsAndSharpnesses[cornerId] = glm::uvec4(friend0, 0, friend1, glm::floatBitsToUint(glm::max(0.f, Sharpness(prevEdgeUnique) - 1.f)));
		}
		newPositions[fp] /= faceSize;
	});

	// --- compute edge-points ---
	std::for_each(EXECUTION_POLICY, edgeMap.cbegin(), edgeMap.cend(), [&](const auto& e) {
		const auto& [edge, entry] = e;
		const auto& [id, a, b] = entry;

		const auto& pa = oldPositions[edge.x];
		const auto& pb = oldPositions[edge.y];

		auto smooth = (pa + pb + newPositions[oV + oE + a.face] + newPositions[oV + oE + b.face]) * .25f;
		auto sharp = (pa + pb) * .5f;

		float sharpness = Sharpness(edge);
		newPositions[oV + id] = glm::mix(smooth, sharp, glm::min(1.f, sharpness));
	});

	// --- update vertex-points ---
	auto vertexView = std::views::iota(0ull, oV);
	std::for_each(EXECUTION_POLICY, vertexView.begin(), vertexView.end(), [&](int v) {
		const auto& oldv = oldPositions[v];

		auto       side = vertexStart[v].load();
		const auto f = side.face;
		const auto slot = side.corner;

		glm::vec3 Q(0, 0, 0);
		glm::vec3 R(0, 0, 0);

		glm::vec3 sharpA;
		glm::vec3 sharpB;

		int   sharpCount = 0;
		float sharpnessSum = 0.f;

		std::size_t n = 0;

		auto f_ = f;
		auto slot_ = slot;
		do {
			n++;

			auto        r = Index(f_, (slot_ + 1) % FaceSize(f_));
			const auto& posE = oldPositions[r];
			R += posE + oldv;
			Q += newPositions[oV + oE + f_];

			auto  edge = UniqueEdge(v, r);
			float sharpness = Sharpness(edge);
			const auto& [id, a, b] = edgeMap[edge];

			sharpnessSum += sharpness;
			if (sharpness > 0) {
				((sharpCount == 0) ? sharpA : sharpB) = posE;
				++sharpCount;
			}

			const auto next = (a.face == f_) ? b : a;
			f_ = next.face;
			slot_ = (next.corner + 1) % FaceSize(f_);
		} while (f_ != f);

		glm::vec3 vertexPoint;
		float     ninv = 1.f / n;

		glm::vec3 smoothRule = ((Q * ninv) + (R * ninv) + (n - 3.f) * oldv) * ninv;
		glm::vec3 creaseRule = sharpA * .125f + oldv * .75f + sharpB * .125f;
		glm::vec3 cornerRule = oldv;

		float vs = sharpnessSum / sharpCount;

		if (sharpCount < 2) {
			vertexPoint = smoothRule;
		} else if (sharpCount > 2) {
			vertexPoint = glm::mix(smoothRule, cornerRule, std::min(vs, 1.f));
		} else {
			vertexPoint = glm::mix(smoothRule, creaseRule, std::min(vs, 1.f));
		}

		newPositions[v] = vertexPoint;
	});

	return EdgefriendGeometry{
		.positions = std::move(newPositions),
		.indices = std::move(newIndices),
		.friendsAndSharpnesses = std::move(newFriendsAndSharpnesses),
		.valenceStartInfos = std::move(newValenceStartInfos)};
}

// We tried to make it easy for you to convert this back to an hlsl shader:
using float3 = glm::vec3;
using int2 = glm::ivec2;
using int4 = glm::ivec4;
using int4x4 = glm::imat4x4;
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;

//constexpr auto asfloat = glm::uintBitsToFloat;
inline auto asfloat(glm::uint32 value) {
	return glm::uintBitsToFloat(value);
}
inline auto asuint(float value) {
	return glm::floatBitsToUint(value);
}
inline auto lerp(const glm::vec3& a, const glm::vec3& b, float value) {
	return glm::mix(a, b, value);
}

glm::uint32_t Load(const auto& buffer, std::uint32_t address) {
	auto bytes = reinterpret_cast<const std::byte*>(buffer.data());
	return *reinterpret_cast<const std::uint32_t*>(bytes + address);
}

glm::uvec2 Load2(const auto& buffer, std::uint32_t address) {
	auto bytes = reinterpret_cast<const std::byte*>(buffer.data());
	return glm::uvec2(*reinterpret_cast<const std::uint32_t*>(bytes + address + 0),
		*reinterpret_cast<const std::uint32_t*>(bytes + address + 4));
}

glm::uvec3 Load3(const auto& buffer, std::uint32_t address) {
	auto bytes = reinterpret_cast<const std::byte*>(buffer.data());
	return glm::uvec3(*reinterpret_cast<const std::uint32_t*>(bytes + address + 0),
		*reinterpret_cast<const std::uint32_t*>(bytes + address + 4),
		*reinterpret_cast<const std::uint32_t*>(bytes + address + 8));
}

glm::uvec4 Load4(const auto& buffer, std::uint32_t address) {
	auto bytes = reinterpret_cast<const std::byte*>(buffer.data());
	return glm::uvec4(*reinterpret_cast<const std::uint32_t*>(bytes + address + 0),
		*reinterpret_cast<const std::uint32_t*>(bytes + address + 4),
		*reinterpret_cast<const std::uint32_t*>(bytes + address + 8),
		*reinterpret_cast<const std::uint32_t*>(bytes + address + 12));
}

const auto Store(auto& buffer, std::uint32_t address, const glm::uint32_t value) {
	auto bytes                                               = reinterpret_cast<std::byte*>(buffer.data());
	*reinterpret_cast<const std::uint32_t*>(bytes + address) = value;
}

const auto Store2(auto& buffer, std::uint32_t address, const glm::uvec2& values) {
	auto bytes                                             = reinterpret_cast<std::byte*>(buffer.data());
	*reinterpret_cast<std::uint32_t*>(bytes + address + 0) = values[0];
	*reinterpret_cast<std::uint32_t*>(bytes + address + 4) = values[1];
}

const auto Store3(auto& buffer, std::uint32_t address, const glm::uvec3& values) {
	auto bytes                                             = reinterpret_cast<std::byte*>(buffer.data());
	*reinterpret_cast<std::uint32_t*>(bytes + address + 0) = values[0];
	*reinterpret_cast<std::uint32_t*>(bytes + address + 4) = values[1];
	*reinterpret_cast<std::uint32_t*>(bytes + address + 8) = values[2];
}

const auto Store4(auto& buffer, std::uint32_t address, const glm::uvec4& values) {
	auto bytes                                              = reinterpret_cast<std::byte*>(buffer.data());
	*reinterpret_cast<std::uint32_t*>(bytes + address + 0)  = values[0];
	*reinterpret_cast<std::uint32_t*>(bytes + address + 4)  = values[1];
	*reinterpret_cast<std::uint32_t*>(bytes + address + 8)  = values[2];
	*reinterpret_cast<std::uint32_t*>(bytes + address + 12) = values[3];
}

void ComputeVertexPoint(
	int vertex,
	const EdgefriendGeometry& old, EdgefriendGeometry& neu) {
	int nFaces = old.friendsAndSharpnesses.size();
	int offset = (vertex > nFaces) ? (3 * nFaces + vertex) : (4 * vertex);

	int corner = old.valenceStartInfos[vertex];
	if (corner < 0 || corner >= nFaces * 4) { // vertex not in use
		neu.valenceStartInfos[offset] = 0x7fffffff;
		neu.positions[offset] = float3(0, 0, 0);
		return;
	}

	neu.valenceStartInfos[offset] = 4 * corner;

	float3 V = old.positions[vertex];
	float3 F = float3(0, 0, 0);
	float3 E = float3(0, 0, 0);

	float3 sharpA;
	float3 sharpB;

	int sharpCount = 0;
	float sharpnessSum = 0.f;

	int corner_ = corner;
	int n = 0;

	do {
		++n;

		int quad = corner_ / 4; // quad id
		int slot = corner_ % 4; // slot inside quad

		int2 EF = int2(Load(old.indices, 4 * (corner_ ^ 3)), Load(old.indices, 4 * (corner_ ^ 2)));

		float3 posE = old.positions[EF.x];

		E += posE;
		F += old.positions[EF.y];

		bool offId = (slot == 0) || (slot == 3);
		uint2 friendAndSharpness = Load2(old.friendsAndSharpnesses, 4 * (4 * quad + 2 * offId));

		corner_ = 2 * friendAndSharpness[0] + (corner_ % 2);

		float sharpness = asfloat(friendAndSharpness[1]);

		sharpnessSum += sharpness;
		if (sharpness > 0) {
			if (sharpCount == 0) {
				sharpA = posE;
			} else {
				sharpB = posE;
			}
			++sharpCount;
		}
	} while (corner_ != corner);

	float beta = 3.f / (2.f * n);
	float gamma = 1.f / (4.f * n);
	float alpha = 1.f - beta - gamma;

	float ni = 1.f / n;

	float3 vertexPoint;

	float3 smoothRule = alpha * V + beta * E * ni + gamma * F * ni;
	float3 creaseRule = sharpA * .125f + V * .75f + sharpB * .125f;
	float3 cornerRule = V;

	float vs = sharpnessSum / sharpCount;

	if (sharpCount < 2) {
		vertexPoint = smoothRule;
	} else if (sharpCount > 2) {
		vertexPoint = lerp(smoothRule, cornerRule, glm::min(vs, 1.f));
	} else {
		vertexPoint = lerp(smoothRule, creaseRule, glm::min(vs, 1.f));
	}

	neu.positions[offset] = vertexPoint;
}

void CSEdgefriend(uint3 dispatchThreadID, const EdgefriendGeometry& old, EdgefriendGeometry& neu) {
	int vertex = dispatchThreadID.x;
	if (vertex < old.positions.size()) {
		ComputeVertexPoint(vertex, old, neu);
	}

	int f = dispatchThreadID.x;
	int oF = old.friendsAndSharpnesses.size();
	if (f >= oF) {
		return;
	}

	/*
	B_------B-------A-------A_
	|       |       |       |
	|BCB_C_ 0 ABCD  1 ADD_A_|
	|       |       |       |
	C_------C-------D-------D_

	Facepoint ABCD := (A + B + C + D) / 4

	Edgepoint of offedge BC := (ABCD + BCB_C_ + B + C) / 4
	Edgepoint of offedge DA := (ABCD + ADD_A_ + D + A) / 4
	*/

	uint4 friendsAndSharpness = Load4(old.friendsAndSharpnesses, 4 * 4 * f);

	// --- load left half of quad ---
	int friend0 = friendsAndSharpness[0];
	float sharpness0 = asfloat(friendsAndSharpness[1]);

	int4 indicesBCB_C_ = Load4(old.indices, 4 * 4 * (friend0 / 2));

	int iC = indicesBCB_C_[2 * (friend0 & 1) + 0];
	int iB = indicesBCB_C_[2 * (friend0 & 1) + 1];
	int iB_ = indicesBCB_C_[2 * ((friend0 & 1) ^ 1) + 0];
	int iC_ = indicesBCB_C_[2 * ((friend0 & 1) ^ 1) + 1];

	float3 BC = lerp(old.positions[iB], old.positions[iC], .5f);
	float3 B_C_ = lerp(old.positions[iB_], old.positions[iC_], .5f);

	// --- load right half of quad ---
	int friend1 = friendsAndSharpness[2];
	float sharpness1 = asfloat(friendsAndSharpness[3]);

	int4 indicesADD_A_ = Load4(old.indices, 4 * 4 * (friend1 / 2));

	int iA = indicesADD_A_[2 * (friend1 & 1) + 0];
	int iD = indicesADD_A_[2 * (friend1 & 1) + 1];
	int iD_ = indicesADD_A_[2 * ((friend1 & 1) ^ 1) + 0];
	int iA_ = indicesADD_A_[2 * ((friend1 & 1) ^ 1) + 1];

	float3 DA = lerp(old.positions[iA], old.positions[iD], .5f);
	float3 D_A_ = lerp(old.positions[iD_], old.positions[iA_], .5f);

	// --- compute points ---
	int facePoint = 4 * f + 1;
	int edgePointOn0 = 4 * f + 2;
	int edgePointOn1 = 4 * f + 3;
	int edgePointOff0 = 4 * (friend0 / 2) + 2 + (friend0 % 2);
	int edgePointOff1 = 4 * (friend1 / 2) + 2 + (friend1 % 2);

	neu.positions[facePoint] = lerp(BC, DA, .5f);

	float3 sharpEdgePoint0 = BC;
	float3 sharpEdgePoint1 = DA;

	float3 smoothEdgePoint0 = B_C_ * .125f + BC * .75f + DA * .125f;
	float3 smoothEdgePoint1 = BC * .125f + DA * .75f + D_A_ * .125f;

	float3 edgePoint0 = lerp(smoothEdgePoint0, sharpEdgePoint0, glm::min(1.f, sharpness0));
	float3 edgePoint1 = lerp(smoothEdgePoint1, sharpEdgePoint1, glm::min(1.f, sharpness1));

	neu.positions[edgePointOff0] = edgePoint0;
	neu.positions[edgePointOff1] = edgePoint1;

	// --- compute quad indices ---
	int4 quad = int4(iA, iB, iC, iD);

	int4x4 quads;
	/*
	quad.y-----ON0-------quad.x
	|         |         |
	|    1    |    0    |
	|         |         |
	|         |         |
	OF0--------+--------OF1
	|         |         |
	|    2    |    3    |
	|         |         |
	|         |         |
	quad.z-----ON1-------quad.w
	*/

	quads[0].x = (quad.x > oF) ? (3 * oF + quad.x) : (4 * quad.x);
	quads[0].y = edgePointOn0;
	quads[0].z = facePoint;
	quads[0].w = edgePointOff1;

	quads[1].x = (quad.y > oF) ? (3 * oF + quad.y) : (4 * quad.y);
	quads[1].y = edgePointOff0;
	quads[1].z = facePoint;
	quads[1].w = edgePointOn0;

	quads[2].x = (quad.z > oF) ? (3 * oF + quad.z) : (4 * quad.z);
	quads[2].y = edgePointOn1;
	quads[2].z = facePoint;
	quads[2].w = edgePointOff0;

	quads[3].x = (quad.w > oF) ? (3 * oF + quad.w) : (4 * quad.w);
	quads[3].y = edgePointOff1;
	quads[3].z = facePoint;
	quads[3].w = edgePointOn1;

	int faceId0 = 4 * f + 0;
	int faceId1 = 4 * f + 1;
	int faceId2 = 4 * f + 2;
	int faceId3 = 4 * f + 3;

	float newSharpness0 = glm::max(0.f, sharpness0 - 1.f);
	float newSharpness1 = glm::max(0.f, sharpness1 - 1.f);

	int friendFace0 = 4 * (friend1 / 2) + 2 * (friend1 & 1) + 0;
	uint4 newFriends0;
	newFriends0[0] = 2 * faceId1 + 1;
	newFriends0[1] = 0;
	newFriends0[2] = 2 * friendFace0 + 0;
	newFriends0[3] = asuint(newSharpness1);

	int friendFace1 = 4 * (friend0 / 2) + 2 * (friend0 & 1) + 1;
	uint2 newFriend1;
	newFriend1[0] = 2 * faceId2 + 1;
	newFriend1[1] = 0;

	uint2 newFriend1_;
	newFriend1_[0] = 2 * faceId1 + 0;
	newFriend1_[1] = asuint(newSharpness0);

	int friendFace2 = 4 * (friend0 / 2) + 2 * (friend0 & 1) + 0;
	uint4 newFriends2;
	newFriends2[0] = 2 * faceId3 + 1;
	newFriends2[1] = 0;
	newFriends2[2] = 2 * friendFace2 + 0;
	newFriends2[3] = asuint(newSharpness0);

	int friendFace3 = 4 * (friend1 / 2) + 2 * (friend1 & 1) + 1;
	uint2 newFriend3;
	newFriend3[0] = 2 * faceId0 + 1;
	newFriend3[1] = 0;

	int2 newFriend3_;
	newFriend3_[0] = 2 * faceId3 + 0;
	newFriend3_[1] = asuint(newSharpness1);

	for (int i = 0; i < 4; ++i) {
		Store4(neu.indices, 4 * 4 * (4 * f + i), quads[i]);
	}

	Store4(neu.friendsAndSharpnesses, 4 * 4 * faceId0, newFriends0);

	Store2(neu.friendsAndSharpnesses, 4 * 4 * faceId1, newFriend1);
	Store2(neu.friendsAndSharpnesses, 4 * 4 * friendFace1 + 8, newFriend1_);

	Store4(neu.friendsAndSharpnesses, 4 * 4 * faceId2, newFriends2);

	Store2(neu.friendsAndSharpnesses, 4 * 4 * faceId3, newFriend3);
	Store2(neu.friendsAndSharpnesses, 4 * 4 * friendFace3 + 8, newFriend3_);

	neu.valenceStartInfos[4 * f + 1] = 4 * (4 * f + 0) + 2;

	int fx = 4 * (friend0 / 2);
	int fy = 4 * (friend1 / 2);
	int sx = friend0 % 2;
	int sy = friend1 % 2;

	neu.valenceStartInfos[fx + 2 + sx] = 4 * (fx + 2 * sx) + 1;
	neu.valenceStartInfos[fy + 2 + sy] = 4 * (fy + 2 * sy) + 1;
}

EdgefriendGeometry SubdivideEdgefriendGeometry(const EdgefriendGeometry& old) {
	EdgefriendGeometry neu;
	int oV = old.positions.size();
	neu.positions.resize(oV + 3 * old.valenceStartInfos.size());
	neu.indices.resize(old.indices.size() * 4);
	neu.friendsAndSharpnesses.resize(old.indices.size());
	neu.valenceStartInfos.resize(neu.positions.size());

	auto threadView = std::views::iota(0, oV);
	std::for_each(EXECUTION_POLICY, threadView.begin(), threadView.end(), [&](auto thread) {
		CSEdgefriend(glm::uvec3(thread, 0, 0), old, neu);
	});
	return neu;
}