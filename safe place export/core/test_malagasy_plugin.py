from core.malagasy_plugin import MalagasyPlugin


def main():
    plugin = MalagasyPlugin()

    print("=" * 70)
    print("MALAGASY PLUGIN TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------------
    capabilities = plugin.capabilities()
    assert capabilities["plugin"] == "malagasy_plugin"
    assert capabilities["language"] == "mg"
    assert capabilities["capabilities"]["language_features"] is True

    print("\nCAPABILITIES")
    print(capabilities)

    # --------------------------------------------------------------
    # Supported language forms
    # --------------------------------------------------------------
    supported_languages = [
        "mg",
        "MG",
        "malagasy",
        "Malagasy",
        "fr+mg",
        "mg+fr",
        "en+mg",
        "mg+en",
        "fr/mg",
        "mg,en",
    ]

    unsupported_languages = [
        "fr",
        "en",
        "fr+en",
        "",
        None,
    ]

    print("\nLANGUAGE SUPPORT")
    for language in supported_languages:
        assert plugin.supports(language) is True, language
        print(language, "-> True")

    for language in unsupported_languages:
        assert plugin.supports(language) is False, language
        print(language, "-> False")

    # --------------------------------------------------------------
    # Processing and standardized output
    # --------------------------------------------------------------
    tests = [
        ("mg", "Mahafinaritra be ity horonantsary ity."),
        ("fr+mg", "Cette vidéo dia mahafinaritra."),
        ("en+mg", "This video dia tsara."),
    ]

    print("\nPROCESSING")
    for language, text in tests:
        result = plugin.process(text, language=language)

        assert result["plugin"] == "malagasy_plugin"
        assert result["language"] == "mg"
        assert result["active"] is True
        assert result["input_text"] == text
        assert result["routed_language"] == language
        assert result["vector_shape"][0] == 1
        assert result["active_features"] >= 0
        assert result["vocabulary_size"] > 0

        # Explicitly verify that the plugin does not translate/replace input.
        assert result["input_text"] == text

        print("\nTEST")
        print("language:", language)
        print("supports:", plugin.supports(language))
        print("result:", result)

    # --------------------------------------------------------------
    # Batch transformation
    # --------------------------------------------------------------
    batch = [
        "Mahafinaritra ity.",
        "Tena tsara.",
        "Horonantsary iray.",
    ]
    matrix = plugin.transform_many(batch)
    assert matrix.shape[0] == len(batch)
    assert matrix.shape[1] == plugin.vocabulary_size

    print("\nBATCH TRANSFORM")
    print("shape:", matrix.shape)

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------
    print("\nINPUT VALIDATION")

    for invalid in (None, 123, [], {}):
        try:
            plugin.transform(invalid)
        except TypeError:
            pass
        else:
            raise AssertionError("transform() should reject non-string input")

    try:
        plugin.transform("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("transform() should reject blank input")

    for invalid in (None, "text"):
        try:
            plugin.transform_many(invalid)
        except TypeError:
            pass
        else:
            raise AssertionError("transform_many() should reject invalid input")

    try:
        plugin.transform_many(["ok", 123])
    except TypeError:
        pass
    else:
        raise AssertionError("transform_many() should reject mixed item types")

    print("validation: OK")
    print("\nALL MALAGASY PLUGIN TESTS PASSED")


if __name__ == "__main__":
    main()
