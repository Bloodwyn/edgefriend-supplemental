#include <vector>
#include <unordered_dense.h>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

struct EdgefriendGeometry {
	std::vector<glm::vec3>  positions;
	std::vector<int>        indices;
	std::vector<glm::uvec4> friendsAndSharpnesses;
	std::vector<int>        valenceStartInfos;
};

EdgefriendGeometry SubdivideToEdgefriendGeometry(
	std::vector<glm::vec3> positions,
	std::vector<int> indices,
	std::vector<int> indicesOffsets,
	ankerl::unordered_dense::map<glm::ivec2, float> sharpEdges);

EdgefriendGeometry SubdivideEdgefriendGeometry(const EdgefriendGeometry& old);
