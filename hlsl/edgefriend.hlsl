struct EdgefriendRootConstants
{
    int F; // face count
    int V; // vertex count
};

ConstantBuffer<EdgefriendRootConstants> oldSize : register(b0);

StructuredBuffer<float3> positionBufferIn : register(t0);
ByteAddressBuffer indexBufferIn : register(t1);
ByteAddressBuffer friendAndSharpnessBufferIn : register(t2);
StructuredBuffer<int> valenceStartInfoBufferIn : register(t3);

RWStructuredBuffer<float3> positionBufferOut : register(u0);
RWByteAddressBuffer indexBufferOut : register(u1);
RWByteAddressBuffer friendAndSharpnessBufferOut : register(u2);
RWStructuredBuffer<int> valenceStartInfoBufferOut : register(u3);

void ComputeVertexPoint(int vertex, out int offset, out float3 vertexPoint, out int newValenceStartInfo);

[numthreads(ComputeThreadGroupSize, 1, 1)]
void CSEdgefriend(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int vertex = dispatchThreadID.x;
    if (vertex < oldSize.V)
    {
        int offset;
        float3 vertexPoint;
        int newValenceStartInfo;
        ComputeVertexPoint(vertex, offset, vertexPoint, newValenceStartInfo);
    
        valenceStartInfoBufferOut[offset] = newValenceStartInfo;
        positionBufferOut[offset] = vertexPoint;
    }
   
    int f = dispatchThreadID.x;
    if (f >= oldSize.F)
    {
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

    uint4 friendsAndSharpness = friendAndSharpnessBufferIn.Load4(4 * 4 * f);

    // --- load left half of quad ---
    int friend0 = friendsAndSharpness[0];
    float sharpness0 = asfloat(friendsAndSharpness[1]);
    
    int4 indicesBCB_C_ = indexBufferIn.Load4(4 * 4 * (friend0 / 2));

    int iC = indicesBCB_C_[2 * (friend0 & 1) + 0];
    int iB = indicesBCB_C_[2 * (friend0 & 1) + 1];
    int iB_ = indicesBCB_C_[2 * ((friend0 & 1) ^ 1) + 0];
    int iC_ = indicesBCB_C_[2 * ((friend0 & 1) ^ 1) + 1];

    float3 BC = lerp(positionBufferIn[iB], positionBufferIn[iC], .5f);
    float3 B_C_ = lerp(positionBufferIn[iB_], positionBufferIn[iC_], .5f);

    // --- load right half of quad ---
    int friend1 = friendsAndSharpness[2];
    float sharpness1 = asfloat(friendsAndSharpness[3]);

    int4 indicesADD_A_ = indexBufferIn.Load4(4 * 4 * (friend1 / 2));

    int iA = indicesADD_A_[2 * (friend1 & 1) + 0];
    int iD = indicesADD_A_[2 * (friend1 & 1) + 1];
    int iD_ = indicesADD_A_[2 * ((friend1 & 1) ^ 1) + 0];
    int iA_ = indicesADD_A_[2 * ((friend1 & 1) ^ 1) + 1];

    float3 DA = lerp(positionBufferIn[iA], positionBufferIn[iD], .5f);
    float3 D_A_ = lerp(positionBufferIn[iD_], positionBufferIn[iA_], .5f);

    // --- compute points ---
    int facePoint = 4 * f + 1;
    int edgePointOn0 = 4 * f + 2;
    int edgePointOn1 = 4 * f + 3;
    int edgePointOff0 = 4 * (friend0 / 2) + 2 + (friend0 % 2);
    int edgePointOff1 = 4 * (friend1 / 2) + 2 + (friend1 % 2);

    positionBufferOut[facePoint] = lerp(BC, DA, .5f);

    float3 sharpEdgePoint0 = BC;
    float3 sharpEdgePoint1 = DA;

    float3 smoothEdgePoint0 = B_C_ * .125f + BC * .75f + DA * .125f;
    float3 smoothEdgePoint1 = BC * .125f + DA * .75f + D_A_ * .125f;

    float3 edgePoint0 = lerp(smoothEdgePoint0, sharpEdgePoint0, min(1.f, sharpness0));
    float3 edgePoint1 = lerp(smoothEdgePoint1, sharpEdgePoint1, min(1.f, sharpness1));

    positionBufferOut[edgePointOff0] = edgePoint0;
    positionBufferOut[edgePointOff1] = edgePoint1;

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

    quads[0].x = (quad.x > oldSize.F) ? (3 * oldSize.F + quad.x) : (4 * quad.x);
    quads[0].y = edgePointOn0;
    quads[0].z = facePoint;
    quads[0].w = edgePointOff1;

    quads[1].x = (quad.y > oldSize.F) ? (3 * oldSize.F + quad.y) : (4 * quad.y);
    quads[1].y = edgePointOff0;
    quads[1].z = facePoint;
    quads[1].w = edgePointOn0;

    quads[2].x = (quad.z > oldSize.F) ? (3 * oldSize.F + quad.z) : (4 * quad.z);
    quads[2].y = edgePointOn1;
    quads[2].z = facePoint;
    quads[2].w = edgePointOff0;

    quads[3].x = (quad.w > oldSize.F) ? (3 * oldSize.F + quad.w) : (4 * quad.w);
    quads[3].y = edgePointOff1;
    quads[3].z = facePoint;
    quads[3].w = edgePointOn1;

    int faceId0 = 4 * f + 0;
    int faceId1 = 4 * f + 1;
    int faceId2 = 4 * f + 2;
    int faceId3 = 4 * f + 3;

    float newSharpness0 = max(0.f, sharpness0 - 1.f);
    float newSharpness1 = max(0.f, sharpness1 - 1.f);

    int friendFace0 = 4 * (friend1 / 2) + 2 * (friend1 & 1) + 0;
    int4 newFriends0;
    newFriends0[0] = 2 * faceId1 + 1;
    newFriends0[1] = 0;
    newFriends0[2] = 2 * friendFace0 + 0;
    newFriends0[3] = asint(newSharpness1);

    int friendFace1 = 4 * (friend0 / 2) + 2 * (friend0 & 1) + 1;
    int2 newFriend1;
    newFriend1[0] = 2 * faceId2 + 1;
    newFriend1[1] = 0;

    int2 newFriend1_;
    newFriend1_[0] = 2 * faceId1 + 0;
    newFriend1_[1] = asint(newSharpness0);

    int friendFace2 = 4 * (friend0 / 2) + 2 * (friend0 & 1) + 0;
    int4 newFriends2;
    newFriends2[0] = 2 * faceId3 + 1;
    newFriends2[1] = 0;
    newFriends2[2] = 2 * friendFace2 + 0;
    newFriends2[3] = asint(newSharpness0);

    int friendFace3 = 4 * (friend1 / 2) + 2 * (friend1 & 1) + 1;
    int2 newFriend3;
    newFriend3[0] = 2 * faceId0 + 1;
    newFriend3[1] = 0;

    int2 newFriend3_;
    newFriend3_[0] = 2 * faceId3 + 0;
    newFriend3_[1] = asint(newSharpness1);

    for (int i = 0; i < 4; ++i)
    {
        indexBufferOut.Store4(4 * 4 * (4 * f + i), quads[i]);
    }

    friendAndSharpnessBufferOut.Store4(4 * 4 * faceId0, newFriends0);

    friendAndSharpnessBufferOut.Store2(4 * 4 * faceId1, newFriend1);
    friendAndSharpnessBufferOut.Store2(4 * 4 * friendFace1 + 8, newFriend1_);

    friendAndSharpnessBufferOut.Store4(4 * 4 * faceId2, newFriends2);

    friendAndSharpnessBufferOut.Store2(4 * 4 * faceId3, newFriend3);
    friendAndSharpnessBufferOut.Store2(4 * 4 * friendFace3 + 8, newFriend3_);

    valenceStartInfoBufferOut[4 * f + 1] = 4 * (4 * f + 0) + 2;

    int fx = 4 * (friend0 / 2);
    int fy = 4 * (friend1 / 2);
    int sx = friend0 % 2;
    int sy = friend1 % 2;

    valenceStartInfoBufferOut[fx + 2 + sx] = 4 * (fx + 2 * sx) + 1;
    valenceStartInfoBufferOut[fy + 2 + sy] = 4 * (fy + 2 * sy) + 1;

}

void ComputeVertexPoint(int vertex, out int offset, out float3 vertexPoint, out int newValenceStartInfo)
{
    offset = (vertex > oldSize.F) ? (3 * oldSize.F + vertex) : (4 * vertex);
    
    int corner = valenceStartInfoBufferIn[vertex];
    if (corner < 0 || corner >= oldSize.F * 4)
    { // vertex not in use
        newValenceStartInfo = 0x7fffffff;
        vertexPoint = float3(0, 0, 0);
        return;
    }

    newValenceStartInfo = 4 * corner;

    float3 V = positionBufferIn[vertex];
    float3 F = float3(0, 0, 0);
    float3 E = float3(0, 0, 0);

    float3 sharpA;
    float3 sharpB;

    int sharpCount = 0;
    float sharpnessSum = 0.f;

    int corner_ = corner;
    int n = 0;

    do
    {
        ++n;

        int quad = corner_ / 4; // quad id
        int slot = corner_ % 4; // slot inside quad

        int2 EF = int2(indexBufferIn.Load(4 * (corner_ ^ 3)), indexBufferIn.Load(4 * (corner_ ^ 2)));

        float3 posE = positionBufferIn[EF.x];

        E += posE;
        F += positionBufferIn[EF.y];

        bool offId = (slot == 0) || (slot == 3);
        uint2 friendAndSharpness = friendAndSharpnessBufferIn.Load2(4 * (4 * quad + 2 * offId));

        corner_ = 2 * friendAndSharpness[0] + (corner_ % 2);

        float sharpness = asfloat(friendAndSharpness[1]) * oldSize.sharpnessFactor;
        
        sharpnessSum += sharpness;
        if (sharpness > 0)
        {
            if (sharpCount == 0)
            {
                sharpA = posE;
            }
            else
            {
                sharpB = posE;
            }
            ++sharpCount;
        }
    } while (corner_ != corner);

    

    float beta = 3.f / (2.f * n);
    float gamma = 1.f / (4.f * n);
    float alpha = 1.f - beta - gamma;

    float ni = 1.f / n;

    float3 smoothRule = alpha * V + beta * E * ni + gamma * F * ni;
    float3 creaseRule = sharpA * .125f + V * .75f + sharpB * .125f;
    float3 cornerRule = V;

    float vs = sharpnessSum / sharpCount;

    if (sharpCount < 2)
    {
        vertexPoint = smoothRule;
    }
    else if (sharpCount > 2)
    {
        vertexPoint = lerp(smoothRule, cornerRule, min(vs, 1));
    }
    else
    {
        vertexPoint = lerp(smoothRule, creaseRule, min(vs, 1));
    }
}