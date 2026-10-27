"""One retained-tree budget shared by packages, policy freezes and reports."""

CAP = 250_000_000


def remaining_subtree_cap(output, artifact_root, study):
    if artifact_root is None:
        return CAP
    if not output.resolve().is_relative_to(artifact_root.resolve()):
        raise ValueError('Artifact subtree must be inside the retained measurement tree')
    remaining = CAP - study._artifact_bytes(artifact_root) + study._artifact_bytes(output)
    if remaining < 0:
        raise RuntimeError('Composite output tree exceeds retained artifact cap')
    return remaining


def check_tree(output, study):
    if study._artifact_bytes(output) > CAP:
        raise RuntimeError('Composite output tree exceeds retained artifact cap')
