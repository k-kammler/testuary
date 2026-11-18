"""The small Maven project the Java tests build, and the JDK 21 and Maven they
build it with."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

MAIN = """\
package demo;

public final class Calc {
    private Calc() {
    }

    public static int sign(int x) {
        if (x > 0) {
            return 1;
        }
        if (x < 0) {
            return -1;
        }
        return 0;
    }

    public static int clamp(int x, int lo, int hi) {
        return Math.max(lo, Math.min(hi, x));
    }
}
"""

POM = """\
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>demo</groupId>
  <artifactId>calc</artifactId>
  <version>1.0</version>
  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.release>11</maven.compiler.release>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.14.4</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
"""


def jdk_and_maven():
    """The JDK 21 home that ``LANGADAPTER_TEST_JAVA_HOMES`` lists, the last one
    if it lists several, and the Maven launcher that ``LANGADAPTER_TEST_MAVEN``
    names. The test is skipped when either is missing."""
    homes = {}
    for entry in os.environ.get("LANGADAPTER_TEST_JAVA_HOMES", "").split(os.pathsep):
        release = Path(entry.strip()) / "release"
        if entry.strip() and release.is_file():
            match = re.search(r'JAVA_VERSION="(1\.)?(\d+)',
                              release.read_text(encoding="utf-8", errors="replace"))
            if match:
                homes[int(match.group(2))] = Path(entry.strip())
    maven = os.environ.get("LANGADAPTER_TEST_MAVEN", "").strip()
    if 21 not in homes or not maven or not Path(maven).exists():
        pytest.skip("needs LANGADAPTER_TEST_JAVA_HOMES with a JDK 21 and "
                    "LANGADAPTER_TEST_MAVEN")
    return homes[21], maven
