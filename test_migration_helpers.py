import unittest

from helpers import (
    expanded,
    identifiers,
    is_context_only_question,
    language,
    matching_aliases,
    route_question,
    verified_scope_match,
)


class MigrationHelperTest(unittest.TestCase):
    def test_deterministic_model_and_multilingual_helpers(self):
        self.assertEqual(identifiers("DS-2CE19D3T-AIT3ZF(2.7-13.5mm)"), ["DS-2CE19D3T-AIT3ZF(2.7-13.5MM)"])
        self.assertEqual(language("как добавить палец"), "ru")
        self.assertIn("fingerprint", expanded("как добавить палец"))

    def test_password_reset_is_a_first_class_operation_alias(self):
        query = expanded("Как восстановить пароль на устройстве?")
        self.assertIn("password", query)
        self.assertIn("reset", query)
        self.assertEqual(route_question("Как восстановить пароль на устройстве?"), "operation")

    def test_hik_connect_alias_expands_to_hcserver_retrieval_term(self):
        query = expanded("какой адрес сервера hikconnect")
        self.assertIn("hcserver", query)
        self.assertIn("server address", query)
        self.assertEqual(route_question("какой адрес сервера hikconnect"), "parameter")

    def test_bare_brand_or_platform_is_context_only_but_real_question_is_not(self):
        self.assertTrue(is_context_only_question("iflow"))
        self.assertTrue(is_context_only_question("Hik-Connect"))
        self.assertFalse(is_context_only_question("какой адрес сервера hikconnect"))
        self.assertFalse(is_context_only_question("iflow камера не подключается"))

    def test_bound_alias_selects_knowledge_key_without_becoming_evidence(self):
        aliases = [{"concept": "сброс пароля", "alias": "пароль", "knowledge_key": "password_access.reset"}]
        matches = matching_aliases("Как восстановить пароль?", aliases)
        self.assertEqual({row["knowledge_key"] for row in matches}, {"password_access.reset"})
        self.assertEqual(verified_scope_match("Как восстановить пароль?", {"models": ["DS-K1T320"]}), "unspecified")

    def test_router_covers_mvp_customer_tasks(self):
        self.assertEqual(route_question("Есть ли DS-K1T320 на складе?", ["DS-K1T320"]), "inventory")
        self.assertEqual(route_question("Совместима ли камера с регистратором?", []), "compatibility")
        self.assertEqual(route_question("Почему устройство не видит карту?", []), "fault")
        self.assertEqual(route_question("Как добавить карту пользователю?", []), "operation")
        self.assertEqual(route_question("Сколько каналов у DS-7216HUHI-K2?", ["DS-7216HUHI-K2"]), "parameter")


if __name__ == "__main__":
    unittest.main()
