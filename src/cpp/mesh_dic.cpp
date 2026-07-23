#include "mesh_dic.h"

namespace meshdic {

MeshdicResult solve(
    const std::string& ref_image_path,
    const std::string& def_image_path,
    const std::string& mesh_dir,
    const MeshdicConfig& config)
{
    MeshdicResult result;
    result.converged = false;
    return result;
}

} // namespace meshdic
