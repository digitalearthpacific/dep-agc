import ast
import hashlib
from pathlib import Path
import unittest

from tasks.inputs import canonical


class CanonicalTests(unittest.TestCase):
    def test_extracted_canonical_operations_equal_accepted_54cfce2(self):
        # Frozen LF function-source fingerprints, independent of Python's AST
        # schema version. Processing imports ordinary Python modules.
        expected = {'pin':'1e4a72c748921b82704dcf1b81d4916e94e503f50183c2edb75e781b103a9a76',
            'canonical_vrt':'9083869b38eacdec06d2c660261e99387c1ad24211f6bc3540ce7f684a77cfa9',
            'canonical':'3352360a5661232b0303ba78193e62abee68e9021d82bc2a7bfe341689426bb5'}
        source = Path(canonical.__file__).read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in expected:
                text = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                                 expected.pop(node.name))
        self.assertEqual(expected,{})


if __name__ == '__main__':
    unittest.main()
